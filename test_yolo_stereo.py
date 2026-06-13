import cv2
import time
import yaml
import numpy as np

from camera.camera_manager import CameraManager
from detection.yolo_detector import YoloDetector
from targeting.target_manager import TargetManager
from tracking.tracker import Tracker
from stereo.stereo_processor import StereoProcessor
from targeting.target_selector import TargetSelector
from control.turret_controller import TurretController
from network.image_websocket_server import ImageWebSocketServer


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

target_manager = TargetManager()

target_selector = TargetSelector()

turret_controller = TurretController()

image_server = ImageWebSocketServer()
image_server.start()

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

    center_x = w // 2
    center_y = h // 2

    center_depth = depth_map[
        center_y,
        center_x
    ]

    cv2.circle(
        rect_left,
        (
            center_x,
            center_y
        ),
        5,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        rect_left,
        f"{center_depth:.0f} mm",
        (
            center_x + 10,
            center_y
        ),
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

    targets = target_manager.build_targets(
        tracks,
        depth_map
    )

    active_target = target_selector.select_target(
        targets
    )

    error_x, error_y = (
        turret_controller.calculate_error(
            active_target,
            rect_left.shape[1],
            rect_left.shape[0]
        )
    )

    turret_controller.send_to_pi(
        active_target,
        error_x,
        error_y
    )

    infer_ms = (
        time.time() - t0
    ) * 1000

    for target in targets:

        det = target.detection

        is_active = (
            active_target is not None
            and
            target.track_id == active_target.track_id
        )

        color = (
            (0, 0, 255)
            if is_active
            else
            (0, 255, 0)
        )

        cv2.rectangle(
            rect_left,
            (
                int(det.x1),
                int(det.y1)
            ),
            (
                int(det.x2),
                int(det.y2)
            ),
            color,
            2
        )

        if is_active:

            cv2.putText(
                rect_left,
                "ACTIVE TARGET",
                (
                    int(det.x1),
                    int(det.y1) - 30
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        label = (
            f"ID:{target.track_id} "
            f"{target.class_name} "
            f"{target.distance_mm:.0f}mm"
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
            color,
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
        f"Targets: {len(targets)}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    if error_x is not None:

        cv2.putText(
            rect_left,
            f"Error X: {error_x:.0f}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            rect_left,
            f"Error Y: {error_y:.0f}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.line(
            rect_left,
            (
                rect_left.shape[1] // 2,
                rect_left.shape[0] // 2
            ),
            (
                int(active_target.center_x),
                int(active_target.center_y)
            ),
            (255, 0, 0),
            2
        )

    image_server.send_frame(rect_left)

    cv2.imshow(
        "YOLO + Stereo",
        rect_left
    )

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()