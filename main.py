from network.image_receiver import ImageReceiver
import cv2
import numpy as np

receiver = ImageReceiver(7000)

while True:
    data = receiver.receive_frame()

    image = cv2.imdecode(
        np.frombuffer(data, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        continue

    cv2.imshow("Camera", image)

    if cv2.waitKey(1) == 27:
        break