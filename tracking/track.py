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

        self.track_id = track_id

        self.detection = detection

        self.state = Track.TENTATIVE

        self.age = 1

        self.missed_frames = 0

        self.confirm_count = 1