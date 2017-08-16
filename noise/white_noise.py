import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

rand = np.random.randn(1000)  # random nums

plt.subplot(2, 1, 1)
plt.title("White noise")
plt.plot(rand)

autocorr = sig.fftconvolve(rand, rand[::-1], mode='full')  # fft

plt.subplot(2, 1, 2)
plt.title("Frequence Area")
plt.plot(np.arange(-len(rand) + 1, len(rand)), autocorr)

plt.tight_layout()
plt.show()
