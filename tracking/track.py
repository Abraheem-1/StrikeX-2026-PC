from collections import deque


class Track:

    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2
    DELETED = 3

    def __init__(
        self,
        track_id,
        detection
    ):

        self.distance_history = deque(
            maxlen=10
        )

        self.smoothed_distance_mm = 0

        self.track_id = track_id

        self.detection = detection

        self.state = Track.TENTATIVE

        self.age = 1

        self.missed_frames = 0

        self.confirm_count = 1

    def update_detection(
        self,
        detection
    ):

        self.detection = detection

        self.confirm_count += 1

        self.missed_frames = 0

        if (
            self.confirm_count >= 3
            and
            self.state == Track.TENTATIVE
        ):
            self.state = Track.CONFIRMED

    def update_distance(
        self,
        distance_mm
    ):

        if distance_mm <= 0:
            return

        self.distance_history.append(
            distance_mm
        )

        values = sorted(
            self.distance_history
        )

        middle = len(values) // 2

        self.smoothed_distance_mm = (
            values[middle]
        )