#neuer Versuch
#glm
#20.11.16

import cv2
import numpy as np
import pytesseract
from matplotlib.patches import Shadow
from time import sleep

class CapturePhoto():
    picture1 = ''
    picture2 = ''
    picture3 = ''
    picture4 = ''
    picture5 = ''

    def capture(self, camera):

        self.picture1 = camera.read()[1]
        self.picture1 = cv2.resize(self.picture1, (400, 400))
        print ('CAP')

        self.picture2 = camera.read()[1]
        self.picture2 = cv2.resize(self.picture2, (400, 400))
        print ('CAP')

        self.picture3 = camera.read()[1]
        self.picture3 = cv2.resize(self.picture3, (400, 400))
        print ('CAP')

        self.picture4 = camera.read()[1]
        self.picture4 = cv2.resize(self.picture4, (400, 400))
        print ('CAP')

        self.picture5 = camera.read()[1]
        self.picture5 = cv2.resize(self.picture5, (400, 400))
        print ('CAP')


        cv2.imwrite('picture1.jpg', self.picture1)
        cv2.imwrite('picture2.jpg', self.picture2)
        cv2.imwrite('picture3.jpg', self.picture3)
        cv2.imwrite('picture4.jpg', self.picture4)
        cv2.imwrite('picture5.jpg', self.picture5)




class ShapeDetecter():
    frame = ''
    mask = ''
    cnts = 0
    center = 0
    c = 0
    M = 0
    radius = 0
    x = 0
    y = 0

    booleanFlag = False

    rect_x = ''
    rect_y = ''
    rect_w = ''
    rect_h = ''


    def __init__(self, frame, mask):
        self.frame=frame
        self.mask=mask


    def analyze(self):

        self.cnts = cv2.findContours(self.mask.copy(), cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_SIMPLE)[-2]

        self.center = 0

        ##### to draw all contours

        #cv2.drawContours(self.frame, self.cnts, -1, (0, 255, 0), 3)


        # only proceed if at least one contour was found
        if len(self.cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            self.c = max(self.cnts, key=cv2.contourArea)
            ((self.x, self.y), self.radius) = cv2.minEnclosingCircle(self.c)
            self.M = cv2.moments(self.c)
            self.center = (int(self.M["m10"] / self.M["m00"]), int(self.M["m01"] / self.M["m00"]))

            # only proceed if the radius meets a minimum size
            if self.radius > 20:
                # draw the circle and centroid on the frame,
                cv2.circle(self.frame, (int(self.x), int(self.y)), int(self.radius),
                           (0, 255, 255), 2)
                cv2.circle(self.frame, self.center, 5, (0, 0, 255), -1)

                self.booleanFlag = True




    def analyse2(self):

        #ret, thresh = cv2.threshold(self.mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 5:

             # Find the index of the largest contour
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt = contours[max_index]

            x, y, w, h = cv2.boundingRect(cnt)

            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), -1)

            print(x,y)

        else:
            print "Sorry nothing found"

    def analyse3(self):

        # ret, thresh = cv2.threshold(self.mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 5:

            #for cnt in contours:

                #x, y, w, h = cv2.boundingRect(cnt)
                #cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), -1)

            areaArray = []

            for i, c in enumerate(contours):
                area = cv2.contourArea(c)
                areaArray.append(area)

            sorteddate = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

            largestcontour = sorteddate[0][1]
            secondlagestcontour = sorteddate[1][1]

            x, y, w, h = cv2.boundingRect(largestcontour)

            x_2, y_2, w_2, h_2 = cv2.boundingRect(secondlagestcontour)

            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), -1)
            cv2.rectangle(self.frame, (x_2, y_2), (x_2 + w_2, y_2 + h_2), (0, 0, 255), -1)





        else:
            print "Sorry nothing found"


    def showImg(self, WindowName, which):
        cv2.imshow(WindowName, which)



class RedFilter():
    frame = ''
    hsv = ''
    mask = ''
    res  = ''
    blurred = ''

    #lower_red = np.array([170, 50, 50])
    #upper_red = np.array([180, 255, 255])

    lower_red = np.array([170, 100, 100])
    upper_red = np.array([180, 255, 255])

    def __init__(self, frame):
        self.frame = frame

    def resizeImg(self):
        self.frame = cv2.resize(self.frame, (400, 400))

    def filterRed(self):
        self.hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        self.mask = cv2.inRange(self.hsv, self.lower_red, self.upper_red)
        self.res = cv2.bitwise_and(self.frame, self.frame, mask=self.mask)


    def smothAndBlur(self, which):
        self.blurred = cv2.GaussianBlur(which, (5, 5), 0)

    def showImg(self, WindowName ,which):
        cv2.imshow(WindowName, which)



camera = cv2.VideoCapture(0)

while(1):
    (grabbed, frame) = camera.read()

    redFiler = RedFilter(frame)
    redFiler.resizeImg()
    redFiler.filterRed()

    redFiler.smothAndBlur(redFiler.mask)
    #redFiler.showImg('mask',redFiler.mask)

    redFiler.showImg('blured',redFiler.blurred)

    shapeD = ShapeDetecter(redFiler.frame, redFiler.blurred)
    #shapeD.analyze()
    #shapeD.analyse2()
    shapeD.analyse3()
    shapeD.showImg('detected', shapeD.frame)


    capP = CapturePhoto()

    #if(shapeD.booleanFlag is True):
    #    capP.capture(camera)
    #    break


    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break



