def get_points_along_line(start_position, end_position):
    points = {}
    x1, y1 = start_position
    x2, y2 = end_position
    dx = x2 - x1
    dy = y2 - y1
    steps = max(abs(dx), abs(dy))

    if steps == 0:
        return {1: (x1, y1)}

    for i in range(steps + 1):
        x = round(x1 + i * dx / steps)
        y = round(y1 + i * dy / steps)
        points[i + 1] = (x, y)

    return points


points_dictionary = get_points_along_line((0, 0), (10, 10))
c = 2

