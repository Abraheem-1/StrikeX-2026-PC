import cv2
import time

from camera.camera_manager import CameraManager
from detection.yolo_detector import YoloDetector
from tracking.tracker import Tracker


camera_manager = CameraManager()
tracker = Tracker()

camera_manager.start()

detector = YoloDetector(
    "models/best.pt",
    device="cuda:0"
)


while True:

    frame = camera_manager.get_frame(0)

    if frame is None:
        continue

    t0 = time.time()

    detections = detector.detect(frame)

    tracks = tracker.update(
        detections
    )

    print(
        [
            (
                track.track_id,
                track.state,
                track.confirm_count
            )
            for track in tracks
        ]
    )

    num_detections = len(detections)

    infer_ms = (time.time() - t0) * 1000

    for det in detections:

        print(
            det.class_name,
            f"{det.confidence:.2f}"
        )

        cv2.rectangle(
            frame,
            (int(det.x1), int(det.y1)),
            (int(det.x2), int(det.y2)),
            (0, 255, 0),
            2
        )

        label = f"{det.class_name}"

        cv2.putText(
            frame,
            label,
            (int(det.x1), int(det.y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Inference: {infer_ms:.1f} ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Detections: {num_detections}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "YOLO Test",
        frame
    )

    if cv2.waitKey(1) == 27:
        break