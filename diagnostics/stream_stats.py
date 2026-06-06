import time


class StreamStats:

    def __init__(self):

        self.packet_count = 0
        self.frame_count = 0

        self.fps = 0.0

        self.last_time = time.time()
        self.last_frame_count = 0

        self.last_frame_id = None
        self.dropped_frames = 0

    def packet_received(self):

        self.packet_count += 1

    def frame_received(self, frame_id):

        if self.last_frame_id is not None:

            expected = self.last_frame_id + 1

            if frame_id > expected:

                self.dropped_frames += (
                    frame_id - expected
                )

        self.last_frame_id = frame_id

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

    def get_dropped_frames(self):

        return self.dropped_frames

    def get_packet_count(self):

        return self.packet_count

    def get_frame_count(self):

        return self.frame_count