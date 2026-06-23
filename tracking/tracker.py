from tracking.track import Track
from tracking.utils import center_distance


class Tracker:

    def __init__(self):

        self.tracks = []
        self.next_track_id = 0

        # Gate on how far a track's center may jump between frames.
        # Raised 100 -> 150 to tolerate fast motion + ~174 ms inference lag
        # (the drone travels a real distance between detections). If IDs still
        # split on very fast targets, raise this further, or add velocity
        # prediction to Track (match against the predicted next position).
        self.distance_gate = 150

    def update(self, detections):

        # Age all tracks once per frame
        for track in self.tracks:
            track.missed_frames += 1

        assigned_tracks = set()

        for detection in detections:

            best_track = None
            best_distance = float("inf")

            for track in self.tracks:

                if track.track_id in assigned_tracks:
                    continue

                distance = center_distance(track.detection, detection)

                if distance < best_distance:
                    best_distance = distance
                    best_track = track

            # ── MATCH BY DISTANCE ONLY ────────────────────────────────
            # The old code ALSO required best_iou > 0.30. That split the
            # track every time a fast/laggy box stopped overlapping the
            # previous one (IoU -> 0), spawning a fresh ID per frame — the
            # multi-box behaviour you saw. Center distance already handles
            # motion correctly, so the IoU requirement is removed.
            if best_track is not None and best_distance < self.distance_gate:

                best_track.update_detection(detection)
                best_track.missed_frames = 0          # reset on a successful match
                assigned_tracks.add(best_track.track_id)

            else:
                new_track = Track(self.next_track_id, detection)
                self.tracks.append(new_track)
                self.next_track_id += 1

        # Drop tracks missing too long (coast up to 5 frames)
        self.tracks = [t for t in self.tracks if t.missed_frames < 5]

        return self.tracks