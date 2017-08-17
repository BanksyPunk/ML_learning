import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

n = 10
l = 256
im = np.zeros((l, l))
np.random.seed(1)

points = l * np.random.random((2, n ** 2))
im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1

plt.subplot(3, 2, 1)
plt.imshow(im, cmap=plt.cm.gray)

im = ndimage.gaussian_filter(im, sigma=l / (4. * n))

mask = (im > im.mean()).astype(np.float)
mask += 0.1 * im
img = mask + 0.2 * np.random.randn(*mask.shape)

hist, bin_edges = np.histogram(img, bins=60)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

binary_img = img > 0.5

plt.subplot(3, 2, 2)
plt.imshow(im, cmap=plt.cm.gray)

plt.subplot(3, 2, 3)
plt.imshow(img, cmap=plt.cm.gray)

plt.subplot(3, 2, 4)
plt.imshow(binary_img, cmap=plt.cm.gray)

plt.subplot(3, 2, 5)
plt.plot(hist)
plt.subplot(3, 2, 6)
plt.plot(bin_centers)

plt.show()
