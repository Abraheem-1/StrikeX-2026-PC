from network.image_receiver import ImageReceiver
import cv2
import numpy as np

receiver = ImageReceiver(7001)

while True:

    jpeg_bytes = receiver.receive_frame()

    image = cv2.imdecode(
        np.frombuffer(jpeg_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        continue

    cv2.imshow("Camera1", image)

    if cv2.waitKey(1) == 27:
        break