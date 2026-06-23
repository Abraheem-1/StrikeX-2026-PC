class TargetSelector:

    def __init__(self, switch_margin_mm=150.0):

        # Currently locked active target (by track id)
        self.current_target_id = None

        # A different target must be at least this much closer (mm) than the
        # current one before we switch. Prevents the active target from
        # flipping between boxes every frame on noisy stereo depth.
        self.switch_margin_mm = switch_margin_mm

    def select_target(self, targets):

        if not targets:
            self.current_target_id = None
            return None

        valid_targets = [t for t in targets if t.distance_mm > 0]

        if not valid_targets:
            self.current_target_id = None
            return None

        # Is the currently-locked target still present?
        current = None
        if self.current_target_id is not None:
            for t in valid_targets:
                if t.track_id == self.current_target_id:
                    current = t
                    break

        # Closest candidate this frame
        closest = min(valid_targets, key=lambda t: t.distance_mm)

        if current is None:
            # Lock was lost (target gone) -> acquire the closest
            chosen = closest
        elif closest.distance_mm < current.distance_mm - self.switch_margin_mm:
            # Another target is clearly closer -> switch
            chosen = closest
        else:
            # Stay locked on the current target (hysteresis)
            chosen = current

        self.current_target_id = chosen.track_id
        return chosen