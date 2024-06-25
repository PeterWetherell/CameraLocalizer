from skimage import measure
from imutils import contours
import numpy
#import argparse
import imutils
import cv2 as cv

vid = cv.VideoCapture(3)

def getCamFrame(camera):
    retval,frame=camera.read()
    frame=numpy.rot90(frame)
    return frame

# load the image and convert it to grayscale
image = getCamFrame(vid)
src = image.copy()
#blue_channel = src[:,:,1]
cv.imshow("a",src)#blue_channel)
image = src.copy()#blue_channel.copy()
#gray = image.copy()
#image[:,:,2] = cv.threshold(image[:,:,2], 220, 255, cv.THRESH_BINARY)[1] #10 #numpy.zeros([image.shape[0], image.shape[1]])
cv.imshow("b",image)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) #
gray = cv.GaussianBlur(gray, (1, 1), 0)
cv.imshow("gray", gray)
thresh = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)[1]
thresh = cv.erode(thresh, None, iterations=1)
#thresh = cv.dilate(thresh, None, iterations=3)
#thresh = cv.erode(thresh, None, iterations=3)
cv.imshow("thresh", thresh)
(minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)
#image = orig.copy()
labels = measure.label(thresh, neighbors=8, background=0)
mask = numpy.zeros(thresh.shape, dtype="uint8")
for label in numpy.unique(labels):
	if label == 0:
		continue
	labelMask = numpy.zeros(thresh.shape, dtype="uint8")
	labelMask[labels == label] = 255
	numPixels = cv.countNonZero(labelMask)
	print(numPixels)
	if numPixels > 10:
		mask = cv.add(mask, labelMask)
cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print(len(cnts));
if (len(cnts) > 0):
    cnts = contours.sort_contours(cnts)[0]
    for (i, c) in enumerate(cnts):
            (x, y, w, h) = cv.boundingRect(c)
            ((cX, cY), radius) = cv.minEnclosingCircle(c)
            cv.circle(image, (int(cX), int(cY)), int(radius),
                    (0, 0, 255), 3)
            cv.putText(image, "#{}".format(i + 1), (x, y - 15),
                    cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

cv.imshow("Image", image)
cv.waitKey(0)