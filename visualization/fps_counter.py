import time


class FPSCounter:

    def __init__(self):

        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0.0

    def update(self):

        self.frame_count += 1

        current_time = time.time()

        elapsed = current_time - self.last_time

        if elapsed >= 1.0:

            self.fps = self.frame_count / elapsed

            self.frame_count = 0
            self.last_time = current_time

    def get_fps(self):

        return self.fps