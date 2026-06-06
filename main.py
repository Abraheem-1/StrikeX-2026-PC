from threading import Thread

import cv2
import numpy as np

from network.image_receiver import ImageReceiver
from visualization.frame_store import FrameStore


frame_store0 = FrameStore()
frame_store1 = FrameStore()


def camera_worker(port, frame_store):

    receiver = ImageReceiver(port)

    while True:

        jpeg_bytes = receiver.receive_frame()

        image = cv2.imdecode(
            np.frombuffer(
                jpeg_bytes,
                dtype=np.uint8
            ),
            cv2.IMREAD_COLOR
        )

        if image is None:
            continue

        frame_store.set_frame(image)


camera0_thread = Thread(
    target=camera_worker,
    args=(7000, frame_store0),
    daemon=True
)

camera1_thread = Thread(
    target=camera_worker,
    args=(7001, frame_store1),
    daemon=True
)

camera0_thread.start()
camera1_thread.start()


while True:

    frame0 = frame_store0.get_frame()
    frame1 = frame_store1.get_frame()

    if frame0 is not None:
        cv2.imshow("Camera0", frame0)

    if frame1 is not None:
        cv2.imshow("Camera1", frame1)

    if cv2.waitKey(1) == 27:
        break