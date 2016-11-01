import cv2
import numpy as np

img_org = cv2.imread("IMG3.JPG")
img = cv2.resize(img_org, (400,400))

while(1):

    #Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    #lower_blue = np.array([110,50,50])
    #upper_blue = np.array([130,255,255])
    l_black = np.array([0,0,0])
    u_black = np.array([200,50,100])


    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, l_black, u_black)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= mask)

    cv2.imshow('frame',img)
    cv2.imshow('mask',mask)
    #cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()