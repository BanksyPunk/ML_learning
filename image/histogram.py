import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


def histeq(img, nbr_bins=256):
    """ Histogram equalization of a grayscale image. """
    # 获取直方图p(r)
    imhist1, bins = np.histogram(img.flatten(), nbr_bins, normed=True)
    plt.subplot(2, 2, 3)
    plt.plot(imhist1)
    # 获取T(r)
    cdf = imhist1.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]

    # 获取s，并用s替换原始图像对应的灰度值
    result = np.interp(img.flatten(), bins[:-1], cdf)
    imhist2, bins = np.histogram(result.flatten(), nbr_bins, normed=True)
    plt.subplot(2, 2, 4)
    plt.plot(imhist2)
    return result.reshape(img.shape), cdf


origin = misc.imread("rock.png")
origin = origin[:, :, 0]  # 彩色转灰度图
result, cdf = histeq(origin)

plt.subplot(2, 2, 1)
plt.imshow(origin, cmap=plt.cm.gray)
plt.subplot(2, 2, 2)
plt.imshow(result, cmap=plt.cm.gray)
plt.show()
