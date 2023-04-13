#%%



import matplotlib.pyplot as plt
import numpy as np
# 表格数据
N = [20, 40, 100, 200]
train = np.array([9.8e-3, 0.015, 0.025, 0.03])
test = np.array([0.042, 0.031, 0.014, 0.008])

# 绘制折线图
plt.semilogy(N, train, marker='o', label='Train')
plt.semilogy(N, test, marker='o', label='Test')

# 设置标题、坐标轴标签和图例
plt.title('Data Visualization')
plt.xlabel('N')
plt.ylabel('Values')
plt.legend()

# 显示图表
plt.show()


