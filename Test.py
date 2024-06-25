import cv2 as cv
import time

vid = cv.VideoCapture(0)
t0 = time.clock()
loops = 0
while(True):
    
    print(loops/(time.clock()-t0))
    
    loops += 1
    retVal1, frame = vid.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    retVal2, mask = cv.threshold(gray, 230, 255, cv.THRESH_BINARY)
    
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()