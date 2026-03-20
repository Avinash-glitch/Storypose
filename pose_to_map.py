import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = r"E:\storypose\models\pose_landmarker_lite.task"
INPUT_IMAGE = r"E:\storypose\kid_pose.jpg"
OUTPUT_IMAGE = r"E:\storypose\pose_map.png"

# MediaPipe landmark indices we care about
POSE_CONNECTIONS = [
    (11, 12),  # shoulders
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 23), (12, 24),  # torso
    (23, 24),  # hips
    (23, 25), (25, 27),  # left leg
    (24, 26), (26, 28),  # right leg
]

def main():
    image_bgr = cv2.imread(INPUT_IMAGE)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {INPUT_IMAGE}")

    h, w = image_bgr.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        raise RuntimeError("No pose detected.")

    landmarks = result.pose_landmarks[0]

    points = []
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))

    # Draw joints
    for x, y in points:
        cv2.circle(canvas, (x, y), 4, (255, 255, 255), -1)

    # Draw skeleton
    for a, b in POSE_CONNECTIONS:
        if a < len(points) and b < len(points):
            cv2.line(canvas, points[a], points[b], (255, 255, 255), 3)

    cv2.imwrite(OUTPUT_IMAGE, canvas)
    print(f"Saved pose map to: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()