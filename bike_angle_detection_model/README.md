Bike angle detection model

What you need before running

- You must already have a labeled CSV with:
  - `frame_path`
  - `bike_angle_deg`
- If you used the dataset builder, that CSV is:
  - `create_labeled_dataset/output/synchronized_dataset.csv`

What this does

- Masks bike pixels with YOLO segmentation (background goes black)
- Trains a classifier that predicts bike angle (-180 to 180) with circular soft labels

1) Preprocess

```bash
python 1_preprocess.py --input_csv ../create_labeled_dataset/output/synchronized_dataset.csv --output_dir data
```

Output:

- `data/dataset.csv`
- `data/frames/` (masked images)

2) Train (best defaults)

```bash
python 2_train.py --data_dir data
```

Model output:

- `models/classification_bins/best_model.pt`

3) Inference

```bash
python 3_inference.py --model models/classification_bins/best_model.pt --image path\\to\\bike.jpg
python 3_inference.py --model models/classification_bins/best_model.pt --csv data\\dataset.csv --output predictions.csv
```

