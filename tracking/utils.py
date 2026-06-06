import math


def center_distance(
    detection_a,
    detection_b
):

    dx = (
        detection_a.center_x
        - detection_b.center_x
    )

    dy = (
        detection_a.center_y
        - detection_b.center_y
    )

    return math.sqrt(
        dx * dx +
        dy * dy
    )