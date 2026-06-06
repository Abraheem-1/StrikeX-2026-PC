import cv2

from camera.camera_stream import CameraStream


camera0 = CameraStream(7000)
camera1 = CameraStream(7001)

camera0.start()
camera1.start()


while True:

    frame0 = camera0.get_frame()
    frame1 = camera1.get_frame()

    if frame0 is not None:

        cv2.putText(
            frame0,
            f"FPS:{camera0.get_fps():.1f}  Drop:{camera0.get_dropped_frames()}  JPEG:{camera0.get_jpeg_size()}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Camera0", frame0)

    if frame1 is not None:

        cv2.putText(
            frame1,
            f"FPS:{camera1.get_fps():.1f}  Drop:{camera1.get_dropped_frames()}  JPEG:{camera1.get_jpeg_size()}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Camera1", frame1)

    if cv2.waitKey(1) == 27:
        break


cv2.destroyAllWindows()