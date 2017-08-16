from scipy.misc import imread, imsave, imresize
from PIL import Image
from pylab import *

img = imread("cat.jpg")
print(img.dtype, img.shape)
img_tinted = img * [1, 0.4, 0.2]
img_tinted = imresize(img_tinted, (300, 300))
imsave('cat_tinted.jpg', img_tinted)

im = array(Image.open('cat.jpg').convert('L'))
figure()
gray()
contour(im, origin='image')
axis('equal')
axis('off')
figure()
hist(im.flatten(),256)
show()