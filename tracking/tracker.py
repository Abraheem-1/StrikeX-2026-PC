from tracking.track import Track


class Tracker:

    def __init__(self):

        self.tracks = []

        self.next_track_id = 0

    def update(
        self,
        detections
    ):

        for detection in detections:

            track = Track(
                self.next_track_id,
                detection
            )

            self.tracks.append(track)

            self.next_track_id += 1

        return self.tracks