import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sign
from scipy import misc

face = misc.ascent()

plt.subplot(1, 2, 1)
plt.title('origin')
plt.imshow(face)

mask = np.array([[-3 - 3j, 0 - 10j, +3 - 3j],
                 [-10 + 0j, 0 + 0j, +10 + 0j],
                 [-3 + 3j, 0 + 10j, +3 + 3j]])

grad = sign.convolve2d(face, mask, boundary='symm', mode='same')

plt.subplot(1, 2, 2)
plt.title('filter')
plt.imshow(np.absolute(grad))

plt.tight_layout()
plt.show()
