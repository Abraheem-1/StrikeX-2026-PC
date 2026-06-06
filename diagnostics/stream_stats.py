import time


class StreamStats:

    def __init__(self):

        self.packet_count = 0
        self.frame_count = 0

        self.fps = 0.0

        self.last_time = time.time()

        self.last_frame_count = 0

    def packet_received(self):

        self.packet_count += 1

    def frame_received(self):

        self.frame_count += 1

        current_time = time.time()

        elapsed = current_time - self.last_time

        if elapsed >= 1.0:

            frames = (
                self.frame_count
                - self.last_frame_count
            )

            self.fps = frames / elapsed

            self.last_frame_count = self.frame_count
            self.last_time = current_time

    def get_fps(self):

        return self.fps