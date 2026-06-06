import cv2

from camera.camera_manager import CameraManager


camera_manager = CameraManager()

camera_manager.start()


while True:

    frame0 = camera_manager.get_frame(0)
    frame1 = camera_manager.get_frame(1)

    if frame0 is not None:

        cv2.putText(
            frame0,
            f"FPS:{camera_manager.get_fps(0):.1f} "
            f"Drop:{camera_manager.get_dropped_frames(0)} "
            f"JPEG:{camera_manager.get_jpeg_size(0)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Camera0", frame0)

    if frame1 is not None:

        cv2.putText(
            frame1,
            f"FPS:{camera_manager.get_fps(1):.1f} "
            f"Drop:{camera_manager.get_dropped_frames(1)} "
            f"JPEG:{camera_manager.get_jpeg_size(1)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Camera1", frame1)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()