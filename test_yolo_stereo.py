import cv2
import time
import yaml
import numpy as np

from camera.camera_manager import CameraManager
from detection.yolo_detector import YoloDetector
from tracking.tracker import Tracker
from stereo.stereo_processor import StereoProcessor


def load_camera_yaml(path):

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    K = np.array(
        data["camera_matrix"]["data"],
        dtype=np.float32
    ).reshape(3, 3)

    D = np.array(
        data["distortion_coefficients"]["data"],
        dtype=np.float32
    )

    R = np.array(
        data["rectification_matrix"]["data"],
        dtype=np.float32
    ).reshape(3, 3)

    P = np.array(
        data["projection_matrix"]["data"],
        dtype=np.float32
    ).reshape(3, 4)

    size = (
        data["image_width"],
        data["image_height"]
    )

    return {
        "K": K,
        "D": D,
        "R": R,
        "P": P,
        "size": size
    }


camera_manager = CameraManager()
camera_manager.start()

tracker = Tracker()

detector = YoloDetector(
    "models/best.pt",
    device="cuda:0"
)

processor = StereoProcessor(
    baseline_mm=80.0,
    rectification_mode="calibrated"
)

left_cam = load_camera_yaml(
    "config/camera0.yaml"
)

right_cam = load_camera_yaml(
    "config/camera1.yaml"
)

processor.set_camera_info(
    K1=left_cam["K"],
    K2=right_cam["K"],
    D1=left_cam["D"],
    D2=right_cam["D"],
    R1=left_cam["R"],
    R2=right_cam["R"],
    P1=left_cam["P"],
    P2=right_cam["P"],
    image_size=left_cam["size"]
)

print(
    "Focal Length:",
    processor.focal_length
)


while True:

    left = camera_manager.get_frame(0)
    right = camera_manager.get_frame(1)

    if left is None or right is None:
        continue

    t0 = time.time()

    rect_left, rect_right = processor.rectify(
        left,
        right
    )

    disparity = processor.compute_disparity(
        rect_left,
        rect_right
    )

    depth_map = processor.compute_depth_mm(
        disparity
    )

    h, w = depth_map.shape

    cx = w // 2
    cy = h // 2

    center_depth = depth_map[cy, cx]

    cv2.circle(
        rect_left,
        (cx, cy),
        5,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        rect_left,
        f"{center_depth:.0f} mm",
        (cx + 10, cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    detections = detector.detect(
        rect_left
    )

    tracks = tracker.update(
        detections
    )

    infer_ms = (
        time.time() - t0
    ) * 1000

    for det in detections:

        cx = int(det.center_x)
        cy = int(det.center_y)

        distance_mm = 0

        if (
            depth_map is not None
            and
            0 <= cy < depth_map.shape[0]
            and
            0 <= cx < depth_map.shape[1]
        ):

            y1 = max(0, cy - 5)
            y2 = min(
                depth_map.shape[0],
                cy + 5
            )

            x1 = max(0, cx - 5)
            x2 = min(
                depth_map.shape[1],
                cx + 5
            )

            region = depth_map[
                y1:y2,
                x1:x2
            ]

            valid = region[
                region > 0
            ]

            if len(valid) > 0:
                distance_mm = int(
                    np.median(valid)
                )

        cv2.rectangle(
            rect_left,
            (int(det.x1), int(det.y1)),
            (int(det.x2), int(det.y2)),
            (0, 255, 0),
            2
        )

        label = (
            f"{det.class_name} "
            f"{distance_mm} mm"
        )

        cv2.putText(
            rect_left,
            label,
            (
                int(det.x1),
                int(det.y1) - 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    cv2.putText(
        rect_left,
        f"Inference: {infer_ms:.1f} ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        rect_left,
        f"Detections: {len(detections)}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "YOLO + Stereo",
        rect_left
    )

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()