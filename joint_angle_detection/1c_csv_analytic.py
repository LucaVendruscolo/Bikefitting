# src/BikeFitting/video_analytic.py

import argparse
from pathlib import Path
import pandas as pd

import cv2
import numpy as np
from ultralytics import YOLO


from core import detect_joints, compute_angles, show_frame_with_data

def parse_args():
    p = argparse.ArgumentParser(description="Joint angle detection from video file")
    p.add_argument("--input", type=str, required=False, default="../bike_angle_detection_model/data/dataset2.csv", help="Path to input CSV dataset")
    p.add_argument("--frames_dir", type=str, required=False, default="../create_labeled_dataset/output/frames", help="Directory containing frame images")
    p.add_argument("--model", type=str, required=False,default="./yolov8m-pose.pt", help="Path to YOLOv8 pose model")
    # Bike angle detection removed
    p.add_argument("--side", type=str, default="auto", choices=["auto", "left", "right"], help="Which side to track")
    p.add_argument("--min_conf", type=float, default=0.5, help="Min keypoint confidence")
    p.add_argument("--output", type=str, default="output/synchronized_dataset.csv", help="Path to save output video (optional)")
    return p.parse_args()


def main():
    args = parse_args()

    input_csv = Path(args.input)
    frames_dir = Path(args.frames_dir)
    if not input_csv.exists():
        print(f"Error: Input CSV not found: {input_csv}")
        return
    if not frames_dir.exists():
        print(f"Error: Frames directory not found: {frames_dir}")
        return

    # Load CSV and model
    df = pd.read_csv(input_csv)
    model = YOLO(args.model)

    updated_rows = []

    for idx, row in df.iterrows():
        frame_path = frames_dir / Path(row['original_frame_path']).name if 'original_frame_path' in row else None
        if frame_path is None or not frame_path.exists():
            print(f"Frame not found for row {idx}: {frame_path}")
            continue
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Could not read frame: {frame_path}")
            continue

        if 'bike_angle_deg' in row:
            bike_angle = row['bike_angle_deg']
        else :
            bike_angle = None

        joints = detect_joints(frame, model, side=args.side, min_conf=args.min_conf)
        angles = compute_angles(joints,bike_angle) if joints is not None else {}

        row_data = {
            'source_video': row['source_video'] if 'source_video' in row else None,
            'frame_path': row['frame_path'] if 'frame_path' in row else None,
            'frame_number': row['frame_number'] if 'frame_number' in row else None,
            'bike_angle_deg': bike_angle,
            'detected_knee_angle': angles.get('knee_angle'),
            'detected_hip_angle': angles.get('hip_angle'),
            'detected_elbow_angle': angles.get('elbow_angle'),
            'detected_crank_angle': angles.get('crank_angle'),
            'foot_x': joints['foot'][0] if joints and 'foot' in joints else None,
            'foot_y': joints['foot'][1] if joints and 'foot' in joints else None,
            'opposite_foot_x': joints['opposite_foot'][0] if joints and 'opposite_foot' in joints else None,
            'opposite_foot_y': joints['opposite_foot'][1] if joints and 'opposite_foot' in joints else None,
            'foot_conf': joints['foot'][2] if joints and 'foot' in joints else None,
            'opposite_foot_conf': joints['opposite_foot'][2] if joints and 'opposite_foot' in joints else None,
        }

        if any(value is None for value in row_data.values()):
            continue

        updated_rows.append(row_data)
        # show frame
        # show_frame_with_data(frame, joints, angles, window_name=f"Frame {idx}")


    updated_df = pd.DataFrame(updated_rows)
    updated_df.to_csv(args.output, index=False)
    print(f"Saved updated CSV to: {args.output}")


if __name__ == "__main__":
    main()
