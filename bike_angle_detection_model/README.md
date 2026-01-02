Bike angle detection model

Predicts camera-relative bike angle (-180 to 180 deg, where 0 = facing camera).

What you need before running

- A labeled CSV with `frame_path` and `bike_angle_deg` columns
- If you used the dataset builder, that CSV is at:
  `create_labeled_dataset/output/synchronized_dataset.csv`

1) Preprocess

Masks bike pixels and subsamples frames.

```bash
python 1_preprocess.py --input_csv ../create_labeled_dataset/output/synchronized_dataset.csv --output_dir data
```

Output: `data/dataset.csv` and `data/frames/` (masked images)

2) Train

```bash
python 2_train.py --data_dir data
```

Default parameters (optimized via Optuna search):
- backbone: convnext_tiny
- num_bins: 120 (3 deg per bin)
- label_smoothing: 22 deg
- lr: 3.4e-5
- batch_size: 48
- epochs: 100

Expected performance: ~2 deg MAE on validation set.

Model saved to: `models/classification_bins/best_model.pt`

3) Inference

```bash
python 3_inference.py --model models/classification_bins/best_model.pt --image path/to/bike.jpg
python 3_inference.py --model models/classification_bins/best_model.pt --csv data/dataset.csv --output predictions.csv
```

Other backbones

You can try other backbones with `--backbone`:
- convnext_tiny (default, best accuracy/speed tradeoff)
- efficientnet_b0 (faster, slightly less accurate)
- resnet50 (classic)
- convnext_small (slower)
