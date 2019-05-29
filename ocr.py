
import pytesseract
from PIL import Image
import cv2 as cv
import numpy as np
import sys

img = cv.imread('payment.jpg',0)
kernel = np.ones((1,1), dtype = "uint8")
img = cv.erode(img, kernel, iterations =1)
img = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
img =cv.medianBlur(img, 3)

cv.imshow('image',img)
cv.waitKey(0)
cv.destryAllWindows()
