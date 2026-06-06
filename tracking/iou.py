def calculate_iou(
    detection_a,
    detection_b
):

    x_left = max(
        detection_a.x1,
        detection_b.x1
    )

    y_top = max(
        detection_a.y1,
        detection_b.y1
    )

    x_right = min(
        detection_a.x2,
        detection_b.x2
    )

    y_bottom = min(
        detection_a.y2,
        detection_b.y2
    )

    if (
        x_right <= x_left
        or
        y_bottom <= y_top
    ):
        return 0.0

    intersection_area = (
        (x_right - x_left)
        *
        (y_bottom - y_top)
    )

    area_a = (
        (detection_a.x2 - detection_a.x1)
        *
        (detection_a.y2 - detection_a.y1)
    )

    area_b = (
        (detection_b.x2 - detection_b.x1)
        *
        (detection_b.y2 - detection_b.y1)
    )

    union_area = (
        area_a
        +
        area_b
        -
        intersection_area
    )

    return (
        intersection_area
        /
        union_area
    )