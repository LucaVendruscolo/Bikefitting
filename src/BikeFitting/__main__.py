# src/BikeFitting/__main__.py

import cv2
from ultralytics import YOLO

from .core import (
    detect_joints,
    compute_angles,
    open_camera,
    show_frame_with_data,
)


def main():
    # load the pose model
    model = YOLO("yolov8n-pose.pt") 

    # get camera input (source 0)
    cap = open_camera(0)
    if not cap.isOpened():
        print("Error: could not open camera (source 0)")
        return

    print("Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No frame from camera, exiting.")
                break

            # 1) image -> joints
            joints = detect_joints(frame, model)

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
