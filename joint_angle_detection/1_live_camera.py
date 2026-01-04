# src/BikeFitting/__main__.py

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from core import detect_joints, compute_angles, open_camera, show_frame_with_data


def parse_args():
    p = argparse.ArgumentParser(description="Live joint angle detection from webcam")
    p.add_argument("--source", type=int, default=0, help="Camera index (default: 0)")
    p.add_argument("--model", type=str, default=None, help="Path to yolov8 pose weights (.pt)")
    p.add_argument("--side", type=str, default="auto", choices=["auto", "left", "right"], help="Which side to track")
    p.add_argument("--min_conf", type=float, default=0.5, help="Min keypoint confidence")
    return p.parse_args()


def main():
    args = parse_args()

    # load the pose model (default: yolov8m-pose for better accuracy)
    default_model = Path(__file__).parent.parent / "models" / "yolov8m-pose.pt"
    model_path = Path(args.model) if args.model else default_model
    
    # Download if not exists
    if not model_path.exists():
        print(f"Model not found at {model_path}, will download yolov8m-pose.pt...")
        model_path = "yolov8m-pose.pt"
    
    model = YOLO(str(model_path))

    # get camera input
    cap = open_camera(args.source)
    if not cap.isOpened():
        print(f"Error: could not open camera (source {args.source})")
        return

    print("Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No frame from camera, exiting.")
                break

            # 1) image -> joints
            joints = detect_joints(frame, model, side=args.side, min_conf=args.min_conf)

            # 2) joints -> angles
            angles = compute_angles(joints) if joints is not None else {}

            # 3 + 4) show live feed with overlay
            show_frame_with_data(frame, joints, angles)

            # quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
