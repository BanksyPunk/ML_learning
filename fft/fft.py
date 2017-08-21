import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 200)  # 创建一个包含30个点的余弦波信号
origin = np.cos(2*x)
plt.subplot(2, 2, 1)
plt.title("origin")
plt.plot(x, origin)

plt.subplot(2, 2, 2)
plt.title("frequency")
transformed = np.fft.fft(origin)  # 使用fft函数对余弦波信号进行傅里叶变换。
plt.plot(transformed)  # 使用Matplotlib绘制变换后的信号。

recover = np.fft.ifft(transformed)
plt.subplot(2, 2, 3)
plt.title("recover")
plt.plot(recover)

plt.subplot(2, 2, 4)
plt.title("delta")
plt.plot(np.fft.ifft(transformed) - origin)

sign = np.all(np.abs(recover - origin) < 10 ** -9)  # 对变换后的结果应用ifft函数，应该可以近似地还原初始信号。

plt.tight_layout()
plt.show()