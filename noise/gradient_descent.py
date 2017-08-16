from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x ** 2 + 10 * np.sin(2 * x) * np.cos(5 * x)


def plot():
    x = np.arange(-10, 10, 0.1)
    plt.plot(x, f(x))
    plt.show()


_global_gradient_descent = optimize.fmin_bfgs(f, 0)
_local = optimize.fmin_bfgs(f, 5)

print(_global_gradient_descent, _local)
plot()
