from tracking.track import Track
from tracking.utils import center_distance


class Tracker:

    def __init__(self):

        self.tracks = []

        self.next_track_id = 0

        self.distance_gate = 100

    def update(
        self,
        detections
    ):

        # Age all tracks once per frame
        for track in self.tracks:

            track.missed_frames += 1

        assigned_tracks = set()

        # Process detections
        for detection in detections:

            best_track = None
            best_distance = float("inf")

            for track in self.tracks:

                distance = center_distance(
                    track.detection,
                    detection
                )

                if distance < best_distance:

                    best_distance = distance
                    best_track = track

            if (
                best_track is not None
                and
                best_distance < self.distance_gate
            ):

                best_track.update_detection(
                    detection
                )

            else:

                new_track = Track(
                    self.next_track_id,
                    detection
                )

                self.tracks.append(
                    new_track
                )

                self.next_track_id += 1

        # Remove tracks that have been missing
        # for 5 frames
        self.tracks = [

            track

            for track in self.tracks

            if track.missed_frames < 5
        ]

        return self.tracks