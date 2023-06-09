# -*- coding: utf-8 -*-
"""AI_HW0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SS7lVLG-471aE2jZYI-wdoYDbSVtNQ8q
"""

import cv2 as cv
import numpy as np
from google.colab.patches import cv2_imshow

img = cv.imread("image.png")

#translation

h, w, c = img.shape
T = np.float32([[1,0,200],
         [0,1,200]])
trans = cv.warpAffine(img, T, (w,h))

cv.imwrite("translated.png", trans)

#rotation
h, w, c = img.shape
R = cv.getRotationMatrix2D((w/2,h/2), 135, 1)

rota = cv.warpAffine(img, R, (w,h))
cv.imwrite("rotation.png", rota)

#flipped
flip = cv.flip(img, 0)
cv.imwrite("flip.png", flip)

#scaling
h, w, c = img.shape
resize = cv.resize(img,(int(0.3*w), int(0.3*h)), interpolation = cv.INTER_NEAREST)

cv.imwrite("scaling.png", resize)

#cropping
#print(img.shape)
crop = img[200:900, 200:900]
cv.imwrite("cropped.png", crop)