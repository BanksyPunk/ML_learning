import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, signal, ndimage

mask = np.array([[-3 - 3j, 0 - 10j, +3 - 3j],
                 [-10 + 0j, 0 + 0j, +10 + 0j],
                 [-3 + 3j, 0 + 10j, +3 + 3j]])

origin = misc.imread("demo.png")

origin = origin[:, :, 0]  # 彩色转灰度图
convolve = signal.convolve2d(origin, mask, boundary='symm', mode='same')  # 拉普拉斯卷积
convolve = np.absolute(convolve)
convolve = convolve > 200 #中值滤波

plt.subplot(2, 2, 1)
plt.imshow(origin, cmap=plt.cm.gray)

plt.subplot(2, 2, 2)
plt.imshow(convolve, cmap=plt.cm.gray, vmin=0, vmax=1)

plt.subplot(2, 2, 3)
plt.imshow(origin - convolve, cmap=plt.cm.gray)

plt.subplot(2, 2, 4)
plt.imshow(origin + convolve, cmap=plt.cm.gray)

plt.show()
