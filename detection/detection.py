class Detection:

    def __init__(
        self,
        class_name,
        confidence,
        x1,
        y1,
        x2,
        y2
    ):
        self.class_name = class_name
        self.confidence = confidence

        self.x1 = x1
        self.y1 = y1

        self.x2 = x2
        self.y2 = y2

    @property
    def center_x(self):
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self):
        return (self.y1 + self.y2) / 2