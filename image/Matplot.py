import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave, imresize

def main():
    # Compute the x and y coordinates for points on sine and cosine curves
    x = np.arange(0, 3 * np.pi, 0.1)
    y_sin = np.sin(x)
    y_cos = np.cos(x)

    # Set up a subplot grid that has height 2 and width 1,
    # and set the first such subplot as active.
    plt.subplot(2, 1, 1)

    # Make the first plot
    plt.plot(x, y_sin)
    plt.title('Sin')

    # Set the second subplot as active, and make the second plot.
    plt.subplot(2, 1, 2)
    plt.plot(x, y_cos)
    plt.title('Cos')

    #Pic
    img = imread('cat.jpg')
    img_tinted = img * [1, 0.55, 0.9]
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.subplot(2, 2, 2)
    # A slight gotcha with imshow is that it might give strange results
    # if presented with data that is not uint8. To work around this, we
    # explicitly cast the image to uint8 before displaying it.
    plt.imshow(np.uint8(img_tinted))

    # Show the figure.
    plt.show()


if __name__ == "__main__":
    main()
