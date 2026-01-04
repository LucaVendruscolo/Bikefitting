# Create Labeled Dataset

This tool syncs video footage with phone IMU (gyroscope) data to create training data for the bike angle model.

## What you need

1. Videos of someone on a bike (filmed from various angles)
2. IMU CSV files from your phone's gyroscope recorded during those videos

## Setup

Put your files in these folders:
- Videos go in: create_labeled_dataset/videos/
- IMU CSVs go in: create_labeled_dataset/imu_runs/

## How to run

```
cd create_labeled_dataset
python 1_build_dataset.py
```

This opens a GUI.

## How to use the GUI

1. Pick a video from the list
2. Scrub through the video to find a frame where your phone's clock is visible
3. Type in the phone time you see
4. Click "Find CSV" - it will auto-match the right IMU file
5. Mark that frame as the sync point
6. Optionally set trim start/end to skip boring parts
7. Save the config
8. Repeat for all your videos
9. Click "Create Dataset"

## Output

After clicking "Create Dataset", you get:
- output/synchronized_dataset.csv - the labels file
- output/frames/ - all the extracted video frames

The CSV has columns:
- frame_path: path to the frame image
- bike_angle_deg: the bike angle (-180 to 180, where 0 = facing camera)

This CSV is what you feed into the bike_angle_detection_model for training.
