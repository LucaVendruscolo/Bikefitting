create_labeled_dataset (dataset builder)

What you need before running

- Put your videos in `create_labeled_dataset/videos/`
  - Example: `create_labeled_dataset/videos/IMG_0811.MOV`
- Put your IMU CSV runs in `create_labeled_dataset/imu_runs/`
  - Example: `create_labeled_dataset/imu_runs/imu_run_20251106_230657.csv`
- `sync_configs.json` is created/updated automatically (you can delete it to reset)

Run

```bash
conda activate bikefitting
cd create_labeled_dataset
python 1_build_dataset.py
```

How it works

- Pick a video
- Find the frame where the phone time is visible
- Type the phone time
- Click “Find CSV” (auto-picks the matching IMU run)
- Mark the sync frame
- Optional: set trim start/end
- Save config for that video
- Repeat, then click “Create Dataset”

Output

- You choose an output folder in the UI (default points at `create_labeled_dataset/output/`)
- It writes:
  - `synchronized_dataset.csv`
  - `frames/` (all extracted frames)

CSV columns you care about

- `frame_path` (relative path to the frame)
- `bike_angle_deg` (target, -180 to 180; 0 = facing camera; ±180 = facing away)
- `sync_time_diff_ms` (sync quality)

This CSV is what the bike angle model uses later.

