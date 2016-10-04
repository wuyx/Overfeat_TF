import sys
sys.path.insert(0, './InitialTraining')
import cv2
import matplotlib.pyplot as plt

from Augment import *

img = cv2.imread('./images/bear1.jpg')

img = resize_image(img, 256, 'min')

img = random_crop(img, (221, 221))

img = flip_image(img)


cv2.imshow('img',img.astype('uint8'))

print img.shape
cv2.waitKey(0)

