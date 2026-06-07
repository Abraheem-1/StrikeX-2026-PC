import numpy as np

from targeting.target import Target


class TargetManager:

    def build_targets(
        self,
        tracks,
        depth_map
    ):

        targets = []

        for track in tracks:

            detection = track.detection

            cx = int(
                detection.center_x
            )

            cy = int(
                detection.center_y
            )

            distance_mm = 0

            if (
                depth_map is not None
                and
                0 <= cy < depth_map.shape[0]
                and
                0 <= cx < depth_map.shape[1]
            ):

                y1 = max(0, cy - 5)
                y2 = min(
                    depth_map.shape[0],
                    cy + 5
                )

                x1 = max(0, cx - 5)
                x2 = min(
                    depth_map.shape[1],
                    cx + 5
                )

                region = depth_map[
                    y1:y2,
                    x1:x2
                ]

                valid = region[
                    region > 0
                ]

                if len(valid) > 0:

                    distance_mm = float(
                        np.median(valid)
                    )

            track.update_distance(
                distance_mm
            )

            target = Target(
                track_id=track.track_id,
                class_name=detection.class_name,
                confidence=detection.confidence,
                distance_mm=track.smoothed_distance_mm,
                detection=detection
            )

            targets.append(
                target
            )

        return targets