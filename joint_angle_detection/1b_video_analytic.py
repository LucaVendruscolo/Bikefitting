# src/BikeFitting/video_analytic.py

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from core import detect_joints, compute_angles, show_frame_with_data


def parse_args():
    p = argparse.ArgumentParser(description="Joint angle detection from video file")
    p.add_argument("--video", type=str, required=True, help="Path to input video file")
    p.add_argument("--model", type=str, default=None, help="Path to yolov8 pose weights (.pt)")
    p.add_argument("--side", type=str, default="auto", choices=["auto", "left", "right"], help="Which side to track")
    p.add_argument("--min_conf", type=float, default=0.5, help="Min keypoint confidence")
    p.add_argument("--output", type=str, default=None, help="Path to save output video (optional)")
    return p.parse_args()


def main():
    args = parse_args()

    # Check video file exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return

    # load the pose model (default: yolov8m-pose for better accuracy)
    default_model = Path(__file__).parent.parent / "models" / "yolov8m-pose.pt"
    model_path = Path(args.model) if args.model else default_model
    
    # Download if not exists
    if not model_path.exists():
        print(f"Model not found at {model_path}, will download yolov8m-pose.pt...")
        model_path = "yolov8m-pose.pt"
    
    model = YOLO(str(model_path))

    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: could not open video file: {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {video_path.name}")
    print(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}")
    print("Press 'q' to quit, 'space' to pause/resume.")

    # Setup video writer if output specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Saving output to: {args.output}")

    frame_num = 0
    paused = False

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    print("End of video.")
                    break
                frame_num += 1

                # 1) image -> joints
                joints = detect_joints(frame, model, side=args.side, min_conf=args.min_conf)

                # 2) joints -> angles
                angles = compute_angles(joints) if joints is not None else {}

                # 3) show frame with overlay
                show_frame_with_data(frame, joints, angles, window_name=f"Video Analysis - {video_path.name}")

                # Write to output if specified
                if writer:
                    # Need to get the visualized frame
                    vis = frame.copy()
                    if joints:
                        for _, (x, y, conf) in joints.items():
                            if conf >= 0.5:
                                cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)
                    writer.write(vis)

                # Show progress
                if frame_num % 30 == 0:
                    print(f"Frame {frame_num}/{total_frames} ({100*frame_num/total_frames:.1f}%)")

            # Handle key presses
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
        print(f"Processed {frame_num} frames.")


if __name__ == "__main__":
    main()
