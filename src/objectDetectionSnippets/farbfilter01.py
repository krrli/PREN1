import cv2
import numpy as np

class Images():
    path=''
    def __init__(self, path):
        self.path=path


class CVStuff():
    img=''
    hsv=''
    mask=''
    res=''

    l_black = np.array([0, 0, 0])
    u_black = np.array([200,50,100])

    def __init__(self, path):
        self.img=cv2.imread(path)

    def resizeImg(self):
        self.img=cv2.resize(self.img, (400,400))

    def filterBlack(self):
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.mask = cv2.inRange(self.hsv, self.l_black, self.u_black)
        self.res = cv2.bitwise_and(self.img,self.img, mask=self.mask)

    def showImg(self):
        cv2.imshow('frame',self.img)
        cv2.imshow('mask',self.mask)

#change path for other JPGs
image=Images("IMG333.JPG")

cvStuff=CVStuff(image.path)
cvStuff.resizeImg()

while(1):
    cvStuff.filterBlack()
    cvStuff.showImg()
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
      break

cv2.destroyAllWindows()
