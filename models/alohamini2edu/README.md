# EDU Model

The body, arms, and lift share current `alohamini2pro` geometry, collision shapes, and joint limits. The C1 lidar uses a 56 × 56 × 42 mm bounding box; dimensions and scan-plane offset from box center are in `config/lidar_collision.json`. Mounting pose is in `config/sensors.yaml`.

After updating the Pro model or nominal lidar mount, synchronize and check from the repository root:

```bash
conda activate alohamini
python scripts/sync_edu_model.py
python -m pytest tests/test_edu_model.py -q
```

Then rebuild the ROS workspace. Device Home calibration and measured sensor extrinsics stay in each machine's workspace.
