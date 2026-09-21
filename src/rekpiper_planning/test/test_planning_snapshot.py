import io
import unittest

import numpy as np
import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from rekpiper_msgs.msg import SDFGrid
from rekpiper_planning.planning_snapshot import PlanningSDFSnapshot


class PlanningSnapshotTest(unittest.TestCase):
    def grid(self):
        return SDFGrid(header=Header(stamp=rospy.Time(100), frame_id='base_link'),
            map_generation_uuid='new-map', bounds_min=Point(0, 0, 0),
            bounds_max=Point(.1, .1, .1), resolution_m=.1,
            size_x=2, size_y=2, size_z=2, distances_m=[-.03] * 8,
            observed=[1] * 7 + [0], valid=False,
            status='stepwise_snapshot_for_planning_preview')

    def test_ros_bytes_unknown_corners_and_outside_are_conservative(self):
        buffer = io.BytesIO()
        self.grid().serialize(buffer)
        grid = SDFGrid().deserialize(buffer.getvalue())
        self.assertIsInstance(grid.observed, bytes)
        snapshot = PlanningSDFSnapshot(grid)
        distances, known = snapshot.query([[0, 0, 0], [.05, .05, .05], [.2, 0, 0]])
        self.assertAlmostEqual(distances[0], -.03)
        self.assertEqual(known.tolist(), [True, False, False])
        self.assertTrue(np.all(distances[1:] >= 1.))
        self.assertFalse(snapshot.sdf_voxels.flags.writeable)
        self.assertFalse(grid.valid)

    def test_mismatched_geometry_provenance_and_flags_are_rejected(self):
        for field, value in [('valid', True), ('status', 'capture_started'),
                             ('map_generation_uuid', ''), ('size_x', 3),
                             ('distances_m', [float('nan')] * 8),
                             ('observed', [1] * 7 + [256])]:
            grid = self.grid()
            setattr(grid, field, value)
            with self.assertRaises(ValueError, msg=field):
                PlanningSDFSnapshot(grid)


if __name__ == '__main__':
    unittest.main()
