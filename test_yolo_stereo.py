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
from live_debug_plot import LiveDebugPlot


def load_camera_yaml(path):

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    K = np.array(data["camera_matrix"]["data"], dtype=np.float32).reshape(3, 3)
    D = np.array(data["distortion_coefficients"]["data"], dtype=np.float32)
    R = np.array(data["rectification_matrix"]["data"], dtype=np.float32).reshape(3, 3)
    P = np.array(data["projection_matrix"]["data"], dtype=np.float32).reshape(3, 4)
    size = (data["image_width"], data["image_height"])

    return {"K": K, "D": D, "R": R, "P": P, "size": size}


camera_manager = CameraManager()
camera_manager.start()

tracker = Tracker()
target_manager = TargetManager()
target_selector = TargetSelector()
turret_controller = TurretController()
debug_plot = LiveDebugPlot()

image_server = ImageWebSocketServer()
image_server.start()

detector = YoloDetector("models/best.pt", device="cuda:0")

processor = StereoProcessor(baseline_mm=80.0, rectification_mode="calibrated")

left_cam = load_camera_yaml("config/camera0.yaml")
right_cam = load_camera_yaml("config/camera1.yaml")

processor.set_camera_info(
    K1=left_cam["K"], K2=right_cam["K"],
    D1=left_cam["D"], D2=right_cam["D"],
    R1=left_cam["R"], R2=right_cam["R"],
    P1=left_cam["P"], P2=right_cam["P"],
    image_size=left_cam["size"]
)

print("Focal Length:", processor.focal_length)


# Lightweight stand-in passed to send_to_pi on the fast path (aiming needs no depth)
class AimTarget:
    def __init__(self, track_id, distance_mm):
        self.track_id = track_id
        self.distance_mm = distance_mm


frame_count = 0
PLOT_EVERY = 10        # redraw the slow matplotlib plot only every Nth frame

while True:

    left = camera_manager.get_frame(0)
    right = camera_manager.get_frame(1)

    if left is None or right is None:
        continue

    frame_count += 1
    t0 = time.time()

    # ================================================================
    # FAST PATH — runs FIRST so the turret gets the freshest error.
    # Aiming is rotation-only and needs NO depth: detect, compute pixel
    # error, send command BEFORE touching stereo.
    # ================================================================
    rect_left, rect_right = processor.rectify(left, right)
    t_rectify = time.time()

    detections = detector.detect(rect_left)
    t_yolo = time.time()

    tracks = tracker.update(detections)

    img_cx = rect_left.shape[1] / 2.0
    img_cy = rect_left.shape[0] / 2.0

    # depth-free active pick: track whose box center is nearest image center
    active_track = None
    if tracks:
        def _center_dist(tr):
            d = tr.detection
            cx = (d.x1 + d.x2) / 2.0
            cy = (d.y1 + d.y2) / 2.0
            return (cx - img_cx) ** 2 + (cy - img_cy) ** 2
        active_track = min(tracks, key=_center_dist)

    center_x = center_y = None
    if active_track is not None:
        d = active_track.detection
        center_x = (d.x1 + d.x2) / 2.0
        center_y = (d.y1 + d.y2) / 2.0
        error_x = center_x - img_cx
        error_y = center_y - img_cy
    else:
        error_x, error_y = None, None

    # fire the command immediately (before stereo)
    if error_x is not None:
        aim = AimTarget(
            active_track.track_id,
            getattr(active_track, "last_distance_mm", 0.0)
        )
        turret_controller.send_to_pi(aim, error_x, error_y)
    t_sent = time.time()

    # ================================================================
    # SLOW PATH — depth AFTER the command is already out.
    # Used for distance display, depth selection, and firing range.
    # ================================================================
    disparity = processor.compute_disparity(rect_left, rect_right)
    depth_map = processor.compute_depth_mm(disparity)
    t_depth = time.time()

    targets = target_manager.build_targets(tracks, depth_map)
    active_target = target_selector.select_target(targets)

    # stash distance back onto each track for next frame's fast-path send
    for tgt in targets:
        for tr in tracks:
            if tr.track_id == tgt.track_id:
                tr.last_distance_mm = tgt.distance_mm
                break
    t_track = time.time()

    # center depth marker (display only)
    h, w = depth_map.shape
    cdx, cdy = w // 2, h // 2
    center_depth = depth_map[cdy, cdx]
    cv2.circle(rect_left, (cdx, cdy), 5, (0, 0, 255), -1)
    cv2.putText(rect_left, f"{center_depth:.0f} mm",
                (cdx + 10, cdy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # slow plot, throttled
    debug_plot.record_error(error_x, error_y)
    if frame_count % PLOT_EVERY == 0:
        debug_plot.update()
    t_plot = time.time()

    # ===== TIMING REPORT — SEND@ is now your real pointing latency =====
    print(
        "rectify={:.0f}  yolo={:.0f}  SEND@={:.0f}  | disp+depth={:.0f}  "
        "sel={:.0f}  plot={:.0f}  | TOTAL={:.0f} ms".format(
            1000 * (t_rectify - t0),
            1000 * (t_yolo - t_rectify),
            1000 * (t_sent - t0),
            1000 * (t_depth - t_sent),
            1000 * (t_track - t_depth),
            1000 * (t_plot - t_track),
            1000 * (t_plot - t0)
        )
    )

    pointing_ms = (t_sent - t0) * 1000

    # ── Display overlays ────────────────────────────────────────────
    for target in targets:
        det = target.detection
        is_active = (active_target is not None
                     and target.track_id == active_target.track_id)
        color = (0, 0, 255) if is_active else (0, 255, 0)

        cv2.rectangle(rect_left,
                      (int(det.x1), int(det.y1)),
                      (int(det.x2), int(det.y2)), color, 2)

        if is_active:
            cv2.putText(rect_left, "ACTIVE TARGET",
                        (int(det.x1), int(det.y1) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        label = f"ID:{target.track_id} {target.class_name} {target.distance_mm:.0f}mm"
        cv2.putText(rect_left, label,
                    (int(det.x1), int(det.y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(rect_left, f"Pointing: {pointing_ms:.1f} ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(rect_left, f"Targets: {len(targets)}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if error_x is not None:
        cv2.putText(rect_left, f"Error X: {error_x:.0f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(rect_left, f"Error Y: {error_y:.0f}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.line(rect_left,
                 (rect_left.shape[1] // 2, rect_left.shape[0] // 2),
                 (int(center_x), int(center_y)),
                 (255, 0, 0), 2)

    image_server.send_frame(rect_left)

    cv2.imshow("YOLO + Stereo", rect_left)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()