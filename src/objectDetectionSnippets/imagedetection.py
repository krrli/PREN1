#apelerkennung von sandro

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    cam = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    colortodetect = [([40, 50, 50], [80, 255, 255])]

    maincolor = 0

    for (lower, upper) in colortodetect:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries and apply the mask
        mask = cv2.inRange(hsv, lower, upper)

        # count #pixel of color
        countnonzero = cv2.countNonZero(mask)

        if maincolor < countnonzero:
            maincolor = countnonzero
            res = cv2.bitwise_and(frame, frame, mask=mask)

        print(maincolor)

        #sensitivity of green detection
        if maincolor > 2000:
            print("Ampel ist gruen")

    cv2.imshow('webcam',mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()