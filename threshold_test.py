import matplotlib
import matplotlib.pyplot as plt

from skimage.data import camera
from skimage.transform import resize
from skimage.filter import threshold_otsu
from skimage.io import imread


matplotlib.rcParams['font.size'] = 9


image = imread('36191.jpg')
image = resize(image, (60,60))
thresh = threshold_otsu(image)
binary = image > (thresh)


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 2.5))
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Original')
ax1.axis('off')

ax2.hist(image)
ax2.set_title('Histogram')
ax2.axvline(thresh, color='r')

ax3.imshow(binary, cmap=plt.cm.gray)
ax3.set_title('Thresholded')
ax3.axis('off')

plt.show()