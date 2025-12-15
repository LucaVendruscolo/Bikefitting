# BikeDataset Builder

A GUI tool to synchronize video frames with IMU sensor data for machine learning training.

## Overview

This tool helps you create a synchronized dataset by:
1. Matching video frames to IMU sensor readings using the phone clock shown at the start of each video
2. Allowing you to trim videos to exclude setup/cleanup periods
3. Extracting frames and combining them with sensor data into a single CSV

## Quick Start

```bash
# Make sure you're in the bikefitting conda environment
conda activate bikefitting

# Run the dataset builder
cd BikeDataset
python dataset_builder.py
```

## Workflow

### Step 1: Select a Video
- Click on a video in the left panel to load it
- Videos marked with ✓ have already been configured

### Step 2: Find the Phone Clock Frame
- Use the playback controls to navigate through the video:
  - **Play/Pause**: Start/stop playback
  - **-10s / +10s**: Jump 10 seconds backward/forward
  - **-1s / +1s**: Jump 1 second backward/forward
  - **-1f / +1f**: Step one frame backward/forward
  - **Slider**: Scrub through the video

### Step 3: Mark the Sync Point
1. Pause on the frame where the phone screen shows the time clearly
2. Enter the time displayed on the phone in the "Phone Time" field
   - Formats accepted: `HH:MM:SS.mmm`, `HH:MM:SS`, or full ISO `YYYY-MM-DDTHH:MM:SS`
3. Click **"🔍 Find CSV"** to automatically find the matching CSV file
   - The tool reads each CSV's time range and finds the one that contains your phone time
   - The matching CSV will be highlighted in the list
4. Click **"📍 Mark Current Frame as Sync Point"**

### Step 4: Verify CSV Match (Automatic)
- The CSV is automatically detected based on the phone time you entered
- The auto-detected CSV is shown below the file list
- You can manually select a different CSV if needed

### Step 5: Trim the Video (Optional)
- Navigate to the first frame you want to include and click **"✂ Set Trim Start"**
- Navigate to the last frame you want to include and click **"✂ Set Trim End"**
- This excludes setup/calibration periods at the start and end

### Step 6: Save Configuration
- Click **"💾 Save This Video's Configuration"**
- The video will be marked with ✓ in the list

### Step 7: Repeat for All Videos
- Configure all your videos following steps 1-6

### Step 8: Create Dataset
- Click **"Create Dataset"** in the left panel
- Select an output directory
- The tool will:
  - Extract all trimmed frames as JPG images
  - Match each frame to the closest IMU reading
  - Create a combined CSV with all data

## Output Format

The tool creates:

```
output_directory/
├── synchronized_dataset.csv    # Combined dataset
└── frames/                     # All extracted video frames
    ├── IMG_0811_frame_000100.jpg
    ├── IMG_0811_frame_000101.jpg
    └── ...
```

### CSV Columns

The output CSV contains:
- `frame_path` - Relative path to the frame image
- `source_video` - Original video filename
- `frame_number` - Frame number in the original video
- `sync_time_diff_ms` - Time difference between frame and IMU reading (ms)
- `wall_time_iso` - Timestamp of the IMU reading
- `euler_heading_deg`, `euler_roll_deg`, `euler_pitch_deg` - Euler angles
- `quat_w`, `quat_x`, `quat_y`, `quat_z` - Quaternion orientation
- `accel_x_m_s2`, `accel_y_m_s2`, `accel_z_m_s2` - Accelerometer data
- `gyro_x_rad_s`, `gyro_y_rad_s`, `gyro_z_rad_s` - Gyroscope data
- And more sensor columns...

## Tips

- **Time Accuracy**: The more precisely you read the phone time, the better the sync
- **Frame Selection**: Pause on a frame where the phone clock digits are clearly visible
- **Trim Generously**: It's better to trim more than needed to avoid noisy data
- **Check Sync Quality**: The `sync_time_diff_ms` column shows how closely each frame matched an IMU reading

## Saved Progress

Your configurations are automatically saved to `sync_configs.json`. You can close the tool and resume later - all configured videos will be remembered.

## Requirements

- Python 3.8+
- OpenCV (cv2)
- Pandas
- Pillow
- tkinter (included with Python)

