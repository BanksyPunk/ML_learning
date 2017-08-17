import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, signal, ndimage

mask = np.array([[-3 - 3j, 0 - 10j, +3 - 3j],
                 [-10 + 0j, 0 + 0j, +10 + 0j],
                 [-3 + 3j, 0 + 10j, +3 + 3j]])

origin = misc.imread("demo.png")
origin = origin[:, :, 0]
signal.fftconvolve()