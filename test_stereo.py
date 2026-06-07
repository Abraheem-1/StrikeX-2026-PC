import cv2
import time
import yaml
import numpy as np

from camera.camera_manager import CameraManager
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

last_print = 0

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

    disparity = processor.compute_disparity(
        left,
        right
    )

    disparity_vis = cv2.normalize(
        disparity,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    disparity_vis = disparity_vis.astype(
        "uint8"
    )

    depth_map = processor.compute_depth_mm(
        disparity
    )

    if depth_map is not None:

        if time.time() - last_print > 1:

            h, w = depth_map.shape

            depth_center = depth_map[
                h // 2,
                w // 2
            ]

            print(
                f"Depth: {depth_center:.1f} mm"
            )

            last_print = time.time()

        depth_vis = cv2.normalize(
            depth_map,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        depth_vis = depth_vis.astype(
            "uint8"
        )

        cv2.imshow(
            "Depth",
            depth_vis
        )

    cv2.imshow(
        "Left",
        left
    )

    cv2.imshow(
        "Right",
        right
    )

    cv2.imshow(
        "Disparity",
        disparity_vis
    )

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()