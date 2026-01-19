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
    p.add_argument("--input", type=str, required=True, help="Path to input video file")
    p.add_argument("--model", type=str, required=False, help="Path to YOLOv8 pose model")
    # Bike angle detection removed
    p.add_argument("--side", type=str, default="auto", choices=["auto", "left", "right"], help="Which side to track")
    p.add_argument("--min_conf", type=float, default=0.5, help="Min keypoint confidence")
    p.add_argument("--output", type=str, default=None, help="Path to save output video (optional)")
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    # Load pose model
    default_model = Path(__file__).parent.parent / "models" / "yolov8m-pose.pt"
    model_path = Path(args.model) if args.model else default_model
    if not model_path.exists():
        print(f"Model not found at {model_path}, will download yolov8m-pose.pt...")
        model_path = "yolov8m-pose.pt"
    model = YOLO(str(model_path))



    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: could not open video file: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Press 'q' to quit, 'space' to pause/resume.")

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Saving output to: {args.output}")

    frame_num = 0
    paused = False
    results_data = []

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    print("End of video.")
                    break
                frame_num += 1

                joints = detect_joints(frame, model, side=args.side, min_conf=args.min_conf)
                angles = compute_angles(joints) if joints is not None else {}
                

                show_frame_with_data(frame, joints, angles, window_name=f"Video Analysis - {input_path.name}")

                row_data = {
                    'source_video': input_path.name,
                    'frame_number': frame_num,
                    'detected_crank_angle': angles.get('crank_angle'),
                    'detected_knee_angle': angles.get('knee_angle'),
                    'detected_hip_angle': angles.get('hip_angle'),
                    'detected_elbow_angle': angles.get('elbow_angle'),
                    'foot_x': joints['foot'][0] if joints and 'foot' in joints else None,
                    'foot_y': joints['foot'][1] if joints and 'foot' in joints else None,
                    'opposite_foot_x': joints['opposite_foot'][0] if joints and 'opposite_foot' in joints else None,
                    'opposite_foot_y': joints['opposite_foot'][1] if joints and 'opposite_foot' in joints else None,
                    'foot_conf': joints['foot'][2] if joints and 'foot' in joints else None,
                    'opposite_foot_conf': joints['opposite_foot'][2] if joints and 'opposite_foot' in joints else None,
                }
                results_data.append(row_data)

                if writer:
                    vis = frame.copy()
                    if joints:
                        for _, (x, y, conf) in joints.items():
                            if conf >= 0.5:
                                cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)
                    writer.write(vis)

                if frame_num % 30 == 0:
                    print(f"Frame {frame_num}/{total_frames} ({100*frame_num/total_frames:.1f}%)")

            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
                print("Paused" if paused else "Resumed")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Save results to CSV
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        output_csv = output_dir / "synchronized_dataset.csv"
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(output_csv, index=False)
        print(f"Saved CSV to: {output_csv}")
        print(f"Processed {frame_num} frames.")


if __name__ == "__main__":
    main()
