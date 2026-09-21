# Deployment contract

The source archive uses one canonical Catkin source space: every ROS package
must be located directly at `Rekpiper/src/<package>`.  A nested
`Rekpiper/src/src/<package>` layout is unsupported and fails the repository
audit.

1. Create an external Ubuntu 20.04/CPython 3.8 x86_64 runtime at
   `REKPIPER_RUNTIME_ROOT` (this checkout uses `runtime/`). Install
   `requirements.lock`, then `requirements.platform.lock`, both with
   `pip install --require-hashes -r <lock>`. Install `requirements.models.lock`
   with `--no-build-isolation --require-hashes` after the locked build tools
   are present. Do not add `--no-deps`. ROS modules remain rosdep/apt
   dependencies. See the [local M0 record](M0_2_TO_M0_5_COMPLETION.md) for
   validated versions, native assets and offline reproduction commands.
2. Create a site-specific, approved replacement for
   `third_party/VENDOR.lock.yaml`. It must contain immutable commits, model
   hashes, CPython ABI and imports for DINOv2, SAM, Cutie, cuRobo, nvblox and
   AnyGrasp. Run `tools/verify_runtime.py` and retain its JSON fingerprint.
3. Verify `third_party/UPSTREAM.lock.yaml` and the fixed ReKep snapshot. Create
   a target-layout source manifest with `tools/create_source_manifest.py`.
4. Build with catkin and inspect `catkin_test_results`; a successful
   `run_tests` process alone is not acceptance.
5. Keep the Ed25519 private key offline. Install only its public key in the
   root-managed trust directory. Sign camera, workspace, gripper, safe-map and
   solver artifacts; schema-v1 files are diagnostic-only and never authorize
   autonomous operation.
   Generic workspace, gripper and safe-map artifacts have mandatory evidence
   content as well as file hashes. The workspace measurement report uses
   `schema_version: 1`, `status: ACCEPTED`, the release `robot_id`, frame,
   bounds and table height. The gripper dataset uses `schema_version: 1`,
   `status: ACCEPTED`, the Piper identity, at least 20 empty cycles, 10 rigid
   grasps and 5 empty/slip trials, plus the exact measured thresholds. Each of
   the five safe-map reports uses `schema_version: 1`, its evidence role as
   `report_type`, `passed: true`, and the release `robot_id`. A signed boolean
   without these matching report contents is rejected.
6. Collect two independent continuous performance records. Each collector run
   has a 30 second discarded warmup and at least 600 measured seconds:
   `--profile software` runs with no Piper hardware graph, while
   `--profile hardware_hold --operator-present --physical-estop-confirmed`
   uses the accepted starting joint vector as an immutable hold target and
   forbids gripper goals. Any drift, invalid map/arm status, discontinuity or
   deadline violation stops the hardware run and produces no passing summary.
   Sign the YAML summary and its immutable JSONL event log separately with
   `approve_runtime_performance.py`.
   For software replay, start `software_performance_sink.launch`, supply the
   fixed replay dataset and executable ReKep program hash to the collector,
   and confirm no Piper node is present. For the real hold test, first create a
   signed `hardware_acceptance_bundle` with `create_release_bundle.py
   --bundle-type hardware_acceptance_bundle`; it contains every final artifact
   except the not-yet-created hardware performance artifact. Start it through
   `start_rekpiper.py --mode autonomous --acceptance-bundle-type
   hardware_acceptance_bundle ... allow_hardware_commands:=true`. This profile
   disables normal closed-loop arming, Cartesian commands, go-zero and all
   gripper commands. The internal bridge and driver independently reject every
   target outside the startup pose. After manual confirmed enable, the
   dedicated hold node emits one immutable 20 Hz hold stream. The signed
   hardware report binds this bootstrap bundle; the final release validator
   rejects any source, URDF, config, robot or artifact drift between the two.
7. Build the final signed `release_bundle.yaml` with
   `create_release_bundle.py`. It binds all eight required signed artifacts,
   official commit, source manifest, URDF, system config, runtime fingerprint,
   robot identity and a monotonic release counter. Install the counter with
   `install_release_counter.py`; rollback counters are rejected.
8. Start production through `start_rekpiper.py`. The offline preflight runs
   before roslaunch; inside the launch graph a supervisor waits for the signed
   preflight result before it creates the Piper SDK or opens CAN. Direct
   `system.launch mode:=shadow` remains available and has no Piper/CAN process.

No bundled tool automatically copies a candidate calibration into an active
site release, enables the arm, or sends a CAN command. This source archive does
not include a real ten-minute performance approval or autonomous release.
