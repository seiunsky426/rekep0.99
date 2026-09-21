#!/usr/bin/env python3
"""Read live M3 topics and save pixel-level evidence; never command hardware."""

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import threading
import time

from cv_bridge import CvBridge
import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import String
import tf2_ros
import yaml

from rekpiper_camera.projection import transform_to_matrix


class ObservationCheck:
    def __init__(self, session, frames):
        self.session = session
        self.frames = frames
        self.workspace = yaml.safe_load((session / "recognition_workspace.yaml").read_text())
        self.lower = np.array(self.workspace["workspace_bounds_min"], dtype=np.float32)
        self.upper = np.array(self.workspace["workspace_bounds_max"], dtype=np.float32)
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.cache = {name: OrderedDict() for name in ("rs1", "rs3")}
        self.results = {name: [] for name in self.cache}
        self.errors = []
        self.fused = []
        self.statuses = []
        self.subscribers = []
        self.syncs = []
        self.sources = json.loads((session / "session.json").read_text())["sources"]
        self.original_transforms = {}
        for name in self.cache:
            source = yaml.safe_load((session / "raw_sources" /
                                     Path(self.sources[name]["source"]).name).read_text())
            self.original_transforms[name] = np.asarray(
                source["calibration"]["base_to_camera_optical"]["matrix_4x4"])
            topics = ["/{}/color/image_raw".format(name),
                      "/{}/aligned_depth_to_color/image_raw".format(name),
                      "/{}/color/camera_info".format(name)]
            subs = [message_filters.Subscriber(topic, msg_type, queue_size=5,
                                               buff_size=16 * 1024 * 1024)
                    for topic, msg_type in zip(topics, (Image, Image, CameraInfo))]
            sync = message_filters.ApproximateTimeSynchronizer(subs, 20, 0.015)
            sync.registerCallback(self.rgbd, name)
            self.syncs.append(sync)
            self.subscribers.extend(subs)
            for key, suffix in (("base", "points_base"), ("roi", "points_recognition")):
                self.subscribers.append(rospy.Subscriber(
                    "/rekpiper/camera/{}/{}".format(name, suffix), PointCloud2,
                    self.cloud, callback_args=(name, key), queue_size=5,
                    buff_size=32 * 1024 * 1024))
        self.subscribers.append(rospy.Subscriber(
            "/rekpiper/camera/fused/points_base", PointCloud2, self.fusion,
            queue_size=10, buff_size=32 * 1024 * 1024))
        self.subscribers.append(rospy.Subscriber(
            "/rekpiper/camera/fused/status", String, self.status, queue_size=30))

    def rgbd(self, rgb, depth, info, name):
        self.store(name, depth.header.stamp.to_nsec(), "rgbd", (rgb, depth, info))

    def cloud(self, message, context):
        name, key = context
        self.store(name, message.header.stamp.to_nsec(), key, message)

    def store(self, name, stamp, key, value):
        with self.lock:
            if len(self.results[name]) >= self.frames:
                return
            cache = self.cache[name]
            cache.setdefault(stamp, {})[key] = value
            while len(cache) > 30:
                cache.popitem(last=False)
            sample = cache.get(stamp, {})
            if set(sample) != {"rgbd", "base", "roi"}:
                return
            del cache[stamp]
            try:
                result = self.check_sample(name, sample)
                self.results[name].append(result)
            except (ValueError, AssertionError, tf2_ros.TransformException) as exc:
                self.errors.append(name + ": " + str(exc))

    @staticmethod
    def xyz(message):
        assert message.header.frame_id == "base_link", "cloud frame"
        assert (message.height, message.width) == (360, 640), "organized HxW"
        assert message.point_step == 12 and message.row_step == 640 * 12, "XYZ stride"
        assert not message.is_bigendian, "XYZ endianness"
        assert [(f.name, f.offset, f.datatype, f.count) for f in message.fields] == [
            ("x", 0, 7, 1), ("y", 4, 7, 1), ("z", 8, 7, 1)], "XYZ fields"
        return np.frombuffer(message.data, dtype="<f4").reshape(360, 640, 3)

    def check_sample(self, name, sample):
        rgb, depth, info = sample["rgbd"]
        stamps = [msg.header.stamp.to_sec() for msg in (rgb, depth, info)]
        assert max(stamps) - min(stamps) <= 0.015, "RGB-D skew"
        assert all(msg.header.frame_id == name + "_color_optical_frame"
                   for msg in (rgb, depth, info)), "optical frame"
        serial = str(rospy.get_param("/" + name + "/realsense2_camera/serial_no"))
        assert serial == self.sources[name]["serial"], "camera serial"
        xyz, roi = self.xyz(sample["base"]), self.xyz(sample["roi"])
        assert sample["base"].header.stamp == depth.header.stamp, "capture stamp"
        assert sample["roi"].header.stamp == depth.header.stamp, "ROI stamp"
        raw = self.bridge.imgmsg_to_cv2(depth, "passthrough")
        scale = 1.0 if depth.encoding == "32FC1" else 0.001
        z = raw.astype(np.float32) * np.float32(scale)
        valid = np.isfinite(z) & (z >= 0.1) & (z <= 2.0)
        assert np.array_equal(np.isfinite(xyz).all(axis=2), valid), "valid pixel indices"
        assert np.isnan(xyz[~valid]).all(), "invalid depth must be all-NaN"
        assert valid.any(), "no valid depth"
        transform = self.buffer.lookup_transform(
            "base_link", name + "_color_optical_frame", depth.header.stamp, rospy.Duration(0.1))
        matrix = transform_to_matrix(transform.transform)
        assert np.allclose(matrix, self.original_transforms[name], atol=1e-6), "TF differs from selected extrinsics"
        # Independent scalar-equivalent projection over distributed pixel samples.
        v, u = np.nonzero(valid)
        selected = np.linspace(0, len(v) - 1, min(128, len(v)), dtype=int)
        v, u = v[selected], u[selected]
        points = np.column_stack(((u-info.K[2])*z[v,u]/info.K[0],
                                  (v-info.K[5])*z[v,u]/info.K[4], z[v,u]))
        expected = points @ matrix[:3, :3].T + matrix[:3, 3]
        error = np.linalg.norm(xyz[v,u] - expected, axis=1)
        assert error.max() < 1e-5, "pixel projection mismatch"
        retained = valid & np.all(xyz >= self.lower, axis=2) & np.all(xyz <= self.upper, axis=2)
        assert np.array_equal(np.isfinite(roi).all(axis=2), retained), "ROI pixel mask"
        assert np.isnan(roi[~retained]).all(), "ROI invalid points must be NaN"
        assert np.array_equal(roi[retained], xyz[retained]), "ROI moved/reordered points"
        assert retained.any(), "ROI empty"
        if not self.results[name]:
            np.savez_compressed(self.session / (name + "_sample.npz"), xyz=xyz, roi=roi,
                                depth=raw, K=np.asarray(info.K), D=np.asarray(info.D),
                                transform=matrix, stamp=stamps[1],
                                rgb=self.bridge.imgmsg_to_cv2(rgb, "rgb8"))
        return dict(stamp=stamps[1], sync_skew_s=max(stamps)-min(stamps),
                    projection_max_error_m=float(error.max()), valid_pixels=int(valid.sum()),
                    roi_pixels=int(retained.sum()), K=list(info.K), D=list(info.D),
                    age_s=rospy.Time.now().to_sec()-stamps[1])

    def fusion(self, message):
        with self.lock:
            try:
                assert message.header.frame_id == "base_link", "fusion frame"
                assert message.height == 1 and message.width > 0, "empty fusion"
                assert message.point_step == 16, "fusion XYZRGB stride"
                data = np.frombuffer(message.data, dtype="<f4").reshape(-1, 4)
                points = data[:, :3]
                assert np.isfinite(points).all(), "nonfinite fused points"
                assert np.all(points >= self.lower-1e-6) and np.all(points <= self.upper+1e-6), "fusion bounds"
                stamp = message.header.stamp.to_sec()
                assert not self.fused or stamp > self.fused[-1]["stamp"], "old fusion stamp returned"
                age = rospy.Time.now().to_sec() - stamp
                assert -0.025 <= age <= 0.6, "fusion is stale on receipt"
                self.fused.append(dict(stamp=stamp, points=message.width, age_s=age))
                if len(self.fused) == 1:
                    np.savez_compressed(self.session / "fused_sample.npz", xyz=points,
                                        rgb=data[:, 3].copy().view(np.uint32), stamp=stamp)
            except (ValueError, AssertionError) as exc:
                self.errors.append("fusion: " + str(exc))

    def status(self, message):
        with self.lock:
            value = json.loads(message.data)
            if value["state"] == "READY":
                if value["stamp_delta_s"] > 0.025:
                    self.errors.append("fusion: source timestamp skew exceeds 25ms")
                if self.statuses and any(
                        value["source_stamps_s"][name] <= self.statuses[-1]["source_stamps_s"][name]
                        for name in ("rs1", "rs3")):
                    self.errors.append("fusion: consumed source frame reused")
                self.statuses.append(value)

    def done(self):
        with self.lock:
            return (all(len(rows) >= self.frames for rows in self.results.values())
                    and len(self.fused) >= 10 and len(self.statuses) >= 10)

    def report(self):
        with self.lock:
            return dict(data_contract_pass=self.done() and not self.errors,
                        physical_acceptance=False, precision_operation_allowed=False,
                        errors=self.errors, requested_frames_per_camera=self.frames,
                        cameras=self.results, fusion=self.fused, fusion_status=self.statuses,
                        workspace=self.workspace, sources=self.sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if args.frames <= 0 or args.timeout <= 0:
        parser.error("frames and timeout must be positive")
    rospy.init_node("m3_observation_check", anonymous=True)
    check = ObservationCheck(args.session, args.frames)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline and not rospy.is_shutdown() and not check.done():
        time.sleep(0.1)
    report = check.report()
    (args.session / "live_observation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(data_contract_pass=report["data_contract_pass"],
                          frames={name: len(rows) for name, rows in report["cameras"].items()},
                          fused_frames=len(report["fusion"]), errors=report["errors"])))
    return 0 if report["data_contract_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
