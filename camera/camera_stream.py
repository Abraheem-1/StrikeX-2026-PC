from diagnostics.stream_stats import StreamStats
from threading import Thread

import cv2
import numpy as np

from network.image_receiver import ImageReceiver
from visualization.frame_store import FrameStore


class CameraStream:

    def __init__(self, port):

        self.receiver = ImageReceiver(port)

        self.frame_store = FrameStore()
        self.stats = StreamStats()
        self.last_jpeg_size = 0

        self.thread = Thread(
            target=self._receiver_loop,
            daemon=True
        )

    def start(self):

        self.thread.start()

    def _receiver_loop(self):

        while True:

            frame_id, jpeg_bytes = (
                self.receiver.receive_frame()
            )
            self.last_jpeg_size = len(jpeg_bytes)
            self.stats.frame_received(frame_id)

            image = cv2.imdecode(
                np.frombuffer(
                    jpeg_bytes,
                    dtype=np.uint8
                ),
                cv2.IMREAD_COLOR
            )

            if image is None:
                continue

            self.frame_store.set_frame(image)

    def get_frame(self):

        return self.frame_store.get_frame()
    
    def get_fps(self):

        return self.stats.get_fps()
    
    def get_dropped_frames(self):
        return self.stats.get_dropped_frames()
    
    def get_jpeg_size(self):
        return self.last_jpeg_size