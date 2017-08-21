import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, signal, ndimage

mask = np.array([[-3 - 3j, 0 - 10j, +3 - 3j],
                 [-10 + 0j, 0 + 0j, +10 + 0j],
                 [-3 + 3j, 0 + 10j, +3 + 3j]])

origin = misc.imread("demo.png")
origin = origin[:, :, 0]  # 彩色转灰度图
origin = np.asarray(origin, dtype="uint8")
print(origin, origin.shape, origin.dtype)
convolve = signal.convolve2d(origin, mask, boundary='symm', mode='same')  # 拉普拉斯卷积
convolve = np.absolute(convolve)
convolve = (convolve > 255) * 255

hist, bin_edges = np.histogram(convolve, bins=255)

plt.subplot(3, 2, 1)
plt.imshow(origin, cmap=plt.cm.gray)

plt.subplot(3, 2, 2)
plt.imshow(convolve, cmap=plt.cm.gray)

plt.subplot(3, 2, 3)
plt.imshow(origin + 0.1 * convolve, cmap=plt.cm.gray)

plt.subplot(3, 2, 4)
plt.plot(hist)

plt.show()
