# src/BikeFitting/core.py

from typing import Dict, Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 / COCO keypoint order (unchanged)
COCO_KPTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
IDX = {name: i for i, name in enumerate(COCO_KPTS)}


# 1) IMAGE -> JOINT COORDINATES (auto side, only 6 joints)
def detect_joints(
    image: np.ndarray,
    model: YOLO,
    side: str = "auto",        # "right", "left" or "auto"
    min_conf: float = 0.5,
) -> Optional[Dict[str, Tuple[float, float, float]]]:
    """
    Takes an image (BGR) and a YOLO pose model.

    Returns a dict of ONLY the joints we care about on ONE side of the body:
        {
          "shoulder": (x, y, conf),
          "elbow":   (x, y, conf),
          "hand":    (x, y, conf),  # YOLO wrist
          "hip":     (x, y, conf),
          "knee":    (x, y, conf),
          "foot":    (x, y, conf),  # YOLO ankle
          "opposite_foot": (x, y, conf),  # YOLO opposite ankle
        }

    side:
      - "right": forced right side
      - "left":  forced left side
      - "auto":  pick whichever side has more visible joints (default)
    """
    side = side.lower()
    if side not in {"right", "left", "auto"}:
        raise ValueError("side must be 'right', 'left' or 'auto'")

    # Helper: mapping from simple names to COCO names for a given side
    def side_map(which: str) -> Dict[str, str]:
        if which == "right":
            return {
                "shoulder": "right_shoulder",
                "elbow": "right_elbow",
                "hand": "right_wrist",
                "hip": "right_hip",
                "knee": "right_knee",
                "foot": "right_ankle",
                "opposite_foot": "left_ankle",
            }
        else:  # "left"
            return {
                "shoulder": "left_shoulder",
                "elbow": "left_elbow",
                "hand": "left_wrist",
                "hip": "left_hip",
                "knee": "left_knee",
                "foot": "left_ankle",
                "opposite_foot": "right_ankle",
            }

    results = model.predict(image, verbose=False)
    if not results or results[0].keypoints is None:
        # no detections at all
        return None

    r = results[0]
    kpts_xy = r.keypoints.xy.cpu().numpy()   # [num_people, 17, 2]
    kpts_cf = r.keypoints.conf.cpu().numpy() # [num_people, 17]

    # if YOLO found zero people, arrays will be empty
    if kpts_cf.size == 0 or kpts_cf.shape[0] == 0:
        return None

    # choose person with highest average keypoint confidence
    person_idx = int(np.argmax(kpts_cf.mean(axis=1)))

    # Function to compute per-side joints + metric
    def joints_for_side(which: str):
        mapping = side_map(which)
        joints_side: Dict[str, Tuple[float, float, float]] = {}
        confidences = []

        for simple_name, coco_name in mapping.items():
            idx = IDX[coco_name]
            x, y = kpts_xy[person_idx, idx]
            c = kpts_cf[person_idx, idx]
            if c < min_conf:
                continue
            joints_side[simple_name] = (float(x), float(y), float(c))
            confidences.append(float(c))

        count = len(confidences)
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        return joints_side, count, mean_conf

    # If side is forced, just use that
    if side in {"right", "left"}:
        joints, count, _ = joints_for_side(side)
        if count == 0:
            return None
        return joints

    # side == "auto": compare left vs right and pick the better one
    joints_right, count_r, mean_r = joints_for_side("right")
    joints_left,  count_l, mean_l = joints_for_side("left")

    # If neither side has enough confidence, return None
    if count_r == 0 and count_l == 0:
        return None

    # Prefer the side with more visible joints; break ties with mean confidence
    if (count_r > count_l) or (count_r == count_l and mean_r >= mean_l):
        return joints_right if joints_right else None
    else:
        return joints_left if joints_left else None


# ============================================================================
# CRANK ANGLE FROM ANKLE POSITIONS
# ============================================================================

def compute_crank_angle(
    near_ankle: Tuple[float, float],
    far_ankle: Tuple[float, float],
) -> float:
    """
    Compute the crank angle from the positions of the near and far ankles.
    
    Convention:
        - 0° = near pedal at 3 o'clock (forward horizontal)
        - 90° = near pedal at 12 o'clock (top dead center)
        - 180° = near pedal at 9 o'clock (back horizontal)
        - 270° = near pedal at 6 o'clock (bottom dead center)
    
    Args:
        near_ankle: (x, y) position of the ankle closer to camera
        far_ankle: (x, y) position of the ankle farther from camera
        
    Returns:
        Crank angle in degrees [0, 360)
    """
    near = np.array(near_ankle)
    far = np.array(far_ankle)
    
    # Crank center is midpoint between the two ankles
    center = (near + far) / 2
    
    # Vector from center to near ankle
    vec = near - center
    
    # Calculate angle using atan2
    # In image coordinates: +x is right, +y is DOWN
    # We negate y to convert to standard math coordinates (y up)
    angle_rad = np.arctan2(-vec[1], vec[0])
    
    # Convert to degrees and normalize to [0, 360)
    angle_deg = np.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    
    return float(angle_deg)


def compute_crank_angle_from_joints(
    joints: Dict[str, Tuple[float, float, float]],
    min_conf: float = 0.7,
) -> Optional[float]:
    """
    Compute crank angle from the joints dict returned by detect_joints.
    
    Args:
        joints: Dict containing "foot" and "opposite_foot" keys
        min_conf: Minimum confidence threshold (default 0.7 = 70%)
        
    Returns:
        Crank angle in degrees [0, 360), or None if ankles not detected
        or confidence is below threshold
    """
    if joints is None:
        return None
    
    if "foot" not in joints or "opposite_foot" not in joints:
        return None
    
    # Check confidence - return None if below 40%
    opp_conf = joints["opposite_foot"][2]
    
    if opp_conf < min_conf:
        #print(f"Warning: Opposite foot confidence {opp_conf:.2f} < {min_conf}, skipping crank detection...")
        return None
    
    near_ankle = (joints["foot"][0], joints["foot"][1])
    far_ankle = (joints["opposite_foot"][0], joints["opposite_foot"][1])

    return compute_crank_angle(near_ankle, far_ankle)


# 2) JOINT POSITIONS -> ANGLES (same 6 joints, single side)
def _angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Return angle ABC in degrees given points a,b,c as (x,y)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return float("nan")
    cosang = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def compute_angles(
    joints: Dict[str, Tuple[float, float, float]]
) -> Dict[str, float]:
    """
    Takes the joint dict from detect_joints and returns a dict of angles
    for this ONE side of the body:

      - "knee_angle"  (hip–knee–foot)
      - "hip_angle"   (shoulder–hip–knee)
      - "elbow_angle" (shoulder–elbow–hand)
      - "crank_angle" (custom formula from ankle positions)
    """

    def pt(name: str) -> Optional[Tuple[float, float]]:
        if name not in joints:
            return None
        x, y, _ = joints[name]
        return (x, y)

    def maybe_angle(a: str, b: str, c: str) -> Optional[float]:
        pa, pb, pc = pt(a), pt(b), pt(c)
        if pa is None or pb is None or pc is None:
            return None
        return _angle(pa, pb, pc)

    angles: Dict[str, float] = {}

    # knee: hip–knee–foot
    k = maybe_angle("hip", "knee", "foot")
    if k is not None:
        angles["knee_angle"] = k

    # hip: shoulder–hip–knee
    h = maybe_angle("shoulder", "hip", "knee")
    if h is not None:
        angles["hip_angle"] = h

    # elbow: shoulder–elbow–hand
    e = maybe_angle("shoulder", "elbow", "hand")
    if e is not None:
        angles["elbow_angle"] = e

    c = compute_crank_angle_from_joints(joints)
    if c is not None:
        angles["crank_angle"] = c

    return angles


# 3) CAMERA INPUT (SOURCE 0)
def open_camera(source: int = 0) -> cv2.VideoCapture:
    """
    Opens the camera with given source (default 0) and returns the VideoCapture.
    """
    cap = cv2.VideoCapture(source)
    return cap


# 4) SHOW LIVE FEED WITH DATA
def show_frame_with_data(
    frame: np.ndarray,
    joints: Optional[Dict[str, Tuple[float, float, float]]],
    angles: Dict[str, float],
    crank_angle: Optional[float] = None,
    window_name: str = "Bike Fitting",
) -> np.ndarray:
    """
    Takes the frame, the joint positions and the angles,
    draws them, shows a live window, and returns the annotated frame.
    """
    vis = frame.copy()

    # Draw joints
    if joints:
        for _, (x, y, conf) in joints.items():
            if conf < 0.5:
                continue
            cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)

    # Helper to place angle text near a joint
    def put_angle_text(angle_key: str, joint_name: str):
        if angle_key not in angles or not joints or joint_name not in joints:
            return
        x, y, _ = joints[joint_name]
        text = f"{angle_key}: {angles[angle_key]:.1f}°"
        cv2.putText(
            vis,
            text,
            (int(x) + 10, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if crank_angle is not None and angle_key == "crank_angle":
            text_crank = f"crank_angle: {crank_angle:.1f}°"
            cv2.putText(
                vis,
                text_crank,
                (int(x) + 10, int(y) + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # We have only one side, so just label near those joints
    put_angle_text("knee_angle", "knee")
    put_angle_text("hip_angle", "hip")
    put_angle_text("elbow_angle", "elbow")
    put_angle_text("crank_angle", "foot")

    cv2.imshow(window_name, vis)
    return vis
