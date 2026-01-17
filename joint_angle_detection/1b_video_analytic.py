import argparse
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from core import detect_joints, compute_angles, show_frame_with_data, compute_crank_angle_from_joints


INPUT_CSV = Path("../create_labeled_dataset/output/synchronized_dataset.csv")
OUTPUT_CSV = Path("output/synchronized_dataset.csv")
OUTPUT_DIR = Path("output")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    df = None
    if INPUT_CSV.exists():
        df = pd.read_csv(INPUT_CSV)
        if video_path.name not in df["source_video"].values:
            df = pd.DataFrame(columns=[
            "source_video", "frame_number",
            "detected_crank_angle", "detected_knee_angle", "detected_hip_angle",
            "detected_elbow_angle", "foot_conf", "opposite_foot_conf",
            "foot_x", "foot_y", "opposite_foot_x", "opposite_foot_y"
        ])
        df["source_video"] = video_path.name
        df["frame_number"] = None
    else:
        for col in ["detected_crank_angle", "detected_knee_angle", "detected_hip_angle", 
                    "detected_elbow_angle", "foot_conf", "opposite_foot_conf",
                    "foot_x", "foot_y", "opposite_foot_x", "opposite_foot_y"]:
            if col not in df.columns:
                    df[col] = None
            print(f"Loaded dataset with {len(df)} rows")

    model_path = Path(__file__).parent.parent / "models" / "yolov8m-pose.pt"
    if not model_path.exists():
        model_path = "yolov8m-pose.pt"
    
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: could not open video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {video_path.name} ({width}x{height}, {fps:.1f}fps, {total_frames} frames)")
    print("'q' to quit, 'space' to pause")

    video_writer = None
    if args.save_video:
        output_video_path = OUTPUT_DIR / f"{video_path.stem}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        print(f"Saving annotated video to: {output_video_path}")

    frame_num = 0
    paused = False

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_num += 1

                joints = detect_joints(frame, model, side="auto", min_conf=0.5)
                angles = compute_angles(joints) if joints is not None else {}
                crank_angle = compute_crank_angle_from_joints(joints, min_conf=0.0)

                mask = (df["source_video"] == video_path.name) & (df["frame_number"] == frame_num)
                if mask.any():
                    df.loc[mask, "detected_crank_angle"] = crank_angle
                    df.loc[mask, "detected_knee_angle"] = angles.get("knee_angle")
                    df.loc[mask, "detected_hip_angle"] = angles.get("hip_angle")
                    df.loc[mask, "detected_elbow_angle"] = angles.get("elbow_angle")
                    foot = joints.get("foot", (None, None, None))
                    opp_foot = joints.get("opposite_foot", (None, None, None))
                    df.loc[mask, "foot_conf"] = foot[2]
                    df.loc[mask, "foot_x"] = foot[0]
                    df.loc[mask, "foot_y"] = foot[1]
                    df.loc[mask, "opposite_foot_conf"] = opp_foot[2]
                    df.loc[mask, "opposite_foot_x"] = opp_foot[0]
                    df.loc[mask, "opposite_foot_y"] = opp_foot[1]
                else:
                    row = {
                        "source_video": video_path.name,
                        "frame_number": frame_num,
                        "detected_crank_angle": crank_angle,
                        "detected_knee_angle": angles.get("knee_angle"),
                        "detected_hip_angle": angles.get("hip_angle"),
                        "detected_elbow_angle": angles.get("elbow_angle"),
                    }
                    if joints is not None:
                        foot = joints.get("foot", (None, None, None))
                        opp_foot = joints.get("opposite_foot", (None, None, None))
                        row.update({
                            "foot_conf": foot[2], "foot_x": foot[0], "foot_y": foot[1],
                            "opposite_foot_conf": opp_foot[2], "opposite_foot_x": opp_foot[0], "opposite_foot_y": opp_foot[1],
                        })
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

                annotated = show_frame_with_data(frame, joints, angles, window_name=f"Video Analysis - {video_path.name}")

                if video_writer is not None:
                    video_writer.write(annotated)

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
        cv2.destroyAllWindows()
        
        if video_writer is not None:
            video_writer.release()
        
        if df is not None:
            df.to_csv(OUTPUT_CSV, index=False)
        
        print(f"Done: {frame_num} frames -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
