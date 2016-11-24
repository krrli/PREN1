#ampelerkennung von sandro

import numpy as np
import cv2

#cap = cv2.VideoCapture(0)



while(True):
    #ret, frame = frame
    frame = cv2.imread('ampel_rot.jpg', 1)

    cam = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    greencolor = [([35, 73, 145], [52, 255, 255])]
    redcolor = [([0, 43, 212], [33, 148, 255])]

    green = 0
    red = 0

    for (lower, upper) in greencolor:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries and apply the mask
        maskgreen = cv2.inRange(hsv, lower, upper)
        resgreen = cv2.bitwise_and(frame, frame, mask = maskgreen)


        # count #pixel of color
        countgreen = cv2.countNonZero(maskgreen)


        if green < countgreen:
            green = countgreen
            resgreen = cv2.bitwise_and(frame, frame, mask=maskred)

        print(green)

        #sensitivity of green detection
        if green > 300:
            print("Ampel ist gruen")


    for (lower, upper) in redcolor:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries and apply the mask
        maskred = cv2.inRange(hsv, lower, upper)
        resred = cv2.bitwise_and(frame, frame, mask = maskred)

        # count #pixel of color
        countred = cv2.countNonZero(maskred)

        if red < countred:
            red = countred
            resred = cv2.bitwise_and(frame, frame, mask=maskred)

        print(red)

        #sensitivity of green detection
        if red > 300:
            print("Ampel ist rot")

    cv2.imshow('maskgreen', maskgreen)
    cv2.imshow('maskred', maskred)
    cv2.imshow('resgreen', resgreen)
    cv2.imshow('resred', resred)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()