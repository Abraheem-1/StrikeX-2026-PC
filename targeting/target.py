class Target:

    def __init__(
        self,
        track_id,
        class_name,
        confidence,
        distance_mm,
        detection
    ):

        self.track_id = track_id
        self.class_name = class_name

        self.confidence = confidence
        self.distance_mm = distance_mm

        self.detection = detection

    @property
    def center_x(self):
        return self.detection.center_x

    @property
    def center_y(self):
        return self.detection.center_y