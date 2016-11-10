#erster Versuch

import cv2
from PIL import Image
import pytesseract
import numpy as np

#print pytesseract.image_to_string(Image.open('roemisch2.PNG'))

#path = './bilder/roemisch2.PNG'
path = 'randomzahlen.jpg'

img = Image.open(path)
print (pytesseract.image_to_string(img))

image = cv2.imread(path)
cv2.imshow('hallo', image)
cv2.waitKey(0)




