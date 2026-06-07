class TargetSelector:

    def select_target(
        self,
        targets
    ):

        if len(targets) == 0:
            return None

        valid_targets = [

            target

            for target in targets

            if target.distance_mm > 0
        ]

        if len(valid_targets) == 0:
            return None

        return min(
            valid_targets,
            key=lambda t: t.distance_mm
        )