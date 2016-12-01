#neuer Versuch
#glm
#20.11.16

import cv2
import numpy as np
import imutils
#import pytesseract
#from matplotlib.patches import Shadow
#from time import sleep
from scipy.spatial import distance as dist


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates

    #TODO
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))-shapeD.w
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))-shapeD.w
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    #captureP = CapturePhoto()
    #captureP.capture(camera)

    # return the warped image
    return warped

class CapturePhoto():
    picture1 = ''
    picture2 = ''
    picture3 = ''
    picture4 = ''
    picture5 = ''


    def capture(self, camera):

        self.picture1 = camera.read()[1]
        self.picture1 = cv2.resize(self.picture1, (400, 400))
        print ('CAP1')

        self.picture2 = camera.read()[1]
        self.picture2 = cv2.resize(self.picture2, (400, 400))
        print ('CAP2')

        self.picture3 = camera.read()[1]
        self.picture3 = cv2.resize(self.picture3, (400, 400))
        print ('CAP3')

        self.picture4 = camera.read()[1]
        self.picture4 = cv2.resize(self.picture4, (400, 400))
        print ('CAP4')

        self.picture5 = camera.read()[1]
        self.picture5 = cv2.resize(self.picture5, (400, 400))
        print ('CAP5')


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
    w = 0
    h = 0

    booleanFlag = False

    rect_x = ''
    rect_y = ''
    rect_w = ''
    rect_h = ''


    def __init__(self, frame, mask):
        self.frame=frame
        self.mask=mask

    def analyse(self):

        # ret, thresh = cv2.threshold(self.mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # >1 --> Test 24.11.16
        if len(contours) > 1:

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

            x, y, self.w, self.h = cv2.boundingRect(largestcontour)

            x_2, y_2, w_2, h_2 = cv2.boundingRect(secondlagestcontour)

            cv2.rectangle(self.frame, (x, y), (x + self.w, y + self.h), (0, 255, 0), -1)
            cv2.rectangle(self.frame, (x_2, y_2), (x_2 + w_2, y_2 + h_2), (0, 0, 255), -1)




            if (x + self.w <= x_2 + w_2 and y + self.h <= y_2 + h_2 and x + self.w >= 10 and y + self.h >= 20):
                print ('CAP')
                self.booleanFlag = True

                a = np.array(largestcontour)
                b = np.array(secondlagestcontour)


                pts = np.vstack((a,b)).squeeze()

                box = cv2.minAreaRect(pts)
                box = cv2.cv.BoxPoints(box)
                box = np.array(box, dtype="int")

                rect = order_points(box)

                print(rect)


                warped = four_point_transform(self.frame, rect)
                cv2.imshow('warped', warped)




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
    shapeD.analyse()
    shapeD.showImg('detected', shapeD.frame)

    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break



