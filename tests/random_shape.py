import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# 设置边界大小
xlim = 10
ylim = 10

# 生成随机点
num_points = 10
x = np.random.randint(xlim, size=num_points)
y = np.random.randint(ylim, size=num_points)

# 连接曲线
t = np.arange(num_points)
spline = make_interp_spline(t, np.column_stack((x, y)), k=3)
x_new, y_new = spline(np.linspace(0, num_points-1, 1000)).T

# 添加随机点
num_random_points = 50
random_points_x = np.random.randint(xlim, size=num_random_points)
random_points_y = np.random.randint(ylim, size=num_random_points)

# 将随机点与曲线点合并成一个多边形
points = np.column_stack((np.concatenate((x_new, random_points_x)), np.concatenate((y_new, random_points_y))))
polygon = plt.Polygon(points, closed=True)

# 绘制多边形
fig, ax = plt.subplots()
ax.add_patch(polygon)
plt.xlim([0, xlim])
plt.ylim([0, ylim])
plt.show()