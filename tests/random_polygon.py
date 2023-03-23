import math, random
from typing import List, Tuple
from PIL import Image
from PIL import ImageDraw


def generate_polygon(center: Tuple[float, float], avg_radius: float,
                     irregularity: float, spikiness: float,
                     num_vertices: int) -> List[Tuple[float, float]]:
    """
    Start with the center of the polygon at center, then creates the
    polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.

    Args:
        center (Tuple[float, float]):
            a pair representing the center of the circumference used
            to generate the polygon.
        avg_radius (float):
            the average radius (distance of each generated vertex to
            the center of the circumference) used to generate points
            with a normal distribution.
        irregularity (float):
            variance of the spacing of the angles between consecutive
            vertices.
        spikiness (float):
            variance of the distance of each vertex to the center of
            the circumference.
        num_vertices (int):
            the number of vertices of the polygon.
    Returns:
        List[Tuple[float, float]]: list of vertices, in CCW order.
    """
    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity)

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        radius = clip(random.gauss(avg_radius, spikiness), 0.1, 2 * avg_radius)
        point = (center[0] + radius * math.cos(angle),
                 center[1] + radius * math.sin(angle))
        points.append(point)
        angle += angle_steps[i]

    return points



def random_angle_steps(steps: int, irregularity: float) -> List[float]:
    """Generates the division of a circumference in random angles.

    Args:
        steps (int):
            the number of angles to generate.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
    Returns:
        List[float]: the list of the random angles.
    """
    # generate n angle steps
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle

    # normalize the steps so that point 0 and point n+1 are the same
    cumsum /= (2 * math.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles


def clip(value, lower, upper):
    """
    Given an interval, values outside the interval are clipped to the interval
    edges.
    """
    return min(upper, max(value, lower))



vertices = generate_polygon(center=(0, 0),
                            avg_radius=1,
                            irregularity=0.3,
                            spikiness=0.4,
                            num_vertices=16)
#
# black = (0, 0, 0)
# white = (255, 255, 255)
# img = Image.new('RGB', (500, 500), white)
# im_px_access = img.load()
# draw = ImageDraw.Draw(img)
#
# # either use .polygon(), if you want to fill the area with a solid colour
# draw.polygon(vertices, outline=black, fill=white)
#
# # or .line() if you want to control the line thickness, or use both methods together!
# draw.line(vertices + [vertices[0]], width=2, fill=black)
#
# img.show()


import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# 创建画布和坐标系
fig, ax = plt.subplots()

# 创建多边形对象并添加到坐标系中
polygon = Polygon(vertices, facecolor='0.9', edgecolor='0.5')
ax.add_patch(polygon)

# 设置坐标轴范围和标题
ax.set_xlim([min(v[0] for v in vertices) - 1, max(v[0] for v in vertices) + 1])
ax.set_ylim([min(v[1] for v in vertices) - 1, max(v[1] for v in vertices) + 1])
ax.set_title('Polygon')
plt.grid()
# 显示图形
plt.show()