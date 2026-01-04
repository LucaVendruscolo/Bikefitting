# Bike Angle Detection Model

Predicts the angle of a bike relative to the camera (-180 to 180 degrees, where 0 = facing you).

## What you need

A labeled CSV with frame_path and bike_angle_deg columns. If you used the dataset builder, this is at:
create_labeled_dataset/output/synchronized_dataset.csv

## Step 1: Preprocess

This masks out everything except the bike in each frame and reduces the frame rate to avoid duplicate training data.

```
python 1_preprocess.py --input_csv ../create_labeled_dataset/output/synchronized_dataset.csv --output_dir data
```

This creates data/dataset.csv and data/frames/ with the masked images.

## Step 2: Train

```
python 2_train.py --data_dir data
```

This trains for about 100 epochs. The best model is saved to the models folder.

Expected accuracy: around 2 degrees average error.

## Step 3: Test on images

Single image:
```
python 3_inference.py --model ../models/best_model.pt --image path/to/bike.jpg
```

Batch predictions:
```
python 3_inference.py --model ../models/best_model.pt --csv data/dataset.csv --output predictions.csv
```

## Generate demo video

There's also a GUI tool to create demo videos with overlaid predictions:

```
python generate_demo_video.py --gui
```

This lets you pick a video, model, and time range, then generates an annotated output video.
