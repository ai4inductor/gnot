#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import numpy as np
import matplotlib.pyplot as plt


z = np.array([[ 0.951,  0.444, -0.903,  0.098,  0.359,
            -0.093,  0.092,  0.297, -0.888,  0.857,
             0.753,  0.201,  0.422,  0.377, -0.559,
             0.763, -0.904, -0.191, -0.630, -0.476,
             0.953,  0.541,  0.868, -0.991,  0.577,
            -0.450,  0.999,  0.872, -0.515,  0.722,
            -0.575,  0.995, -0.890,  0.660, -0.998,
             0.816,  0.446,  0.875, -0.990,  0.616],
       [    -0.306, -0.895,  0.427, -0.995, -0.933,
             0.995, -0.995, -0.954,  0.458,  0.514,
            -0.657, -0.979, -0.906, -0.925,  0.828,
            -0.646,  0.426,  0.981, -0.775, -0.879,
             0.302,  0.840, -0.495, -0.133, -0.816,
            -0.892,  0.038,  0.488,  0.856,  0.690,
             0.817, -0.094,  0.454,  0.750, -0.061,
             0.577, -0.894, -0.482, -0.138,  0.787]])

z1 = np.array([[-0.903,  0.098,  0.359, -0.093,  0.092,  0.297, -0.888,  0.377, -0.559,
  -0.904, -0.191, -0.630, -0.476, -0.991, -0.450, -0.515, -0.575, -0.890,
  -0.998, -0.990],
 [ 0.427, -0.995, -0.933,  0.995, -0.995, -0.954,  0.458, -0.925,  0.828,
   0.426,  0.981, -0.775, -0.879, -0.133, -0.892,  0.856,  0.817,  0.454,
  -0.061, -0.138]]
)


plt.figure()
plt.scatter(z[0],z[1],)
plt.scatter(z1[0],z1[1],s=30)
plt.show()

