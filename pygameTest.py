from skimage import measure
from imutils import contours
import pygame
import cv2 as cv
import numpy
import math
import TextEditorTest
import sys
import os
import time
import refineData
import imutils
import subprocess

class Button(object):
    def __init__(self,X,Y,width,height,displayText):
        self.pressedColor = (170,170,170)
        self.unPressedColor = (100,100,100)
        self.x = X
        self.y = Y
        self.buttonWidth = width
        self.buttonHeight = height
        self.text = displayText
        
    def isTouching(self,mouseX,mouseY):
        return self.x <= mouseX <= self.x+self.buttonWidth and self.y <= mouseY <= self.y+self.buttonHeight
    
    def drawButton(self,screen,mouseX,mouseY):
        font = min(int(self.buttonHeight * 0.75),50)
        smallfont = pygame.font.SysFont('Corbel',font)
        buttonText = smallfont.render(self.text , True , (255,255,255))
        
        if self.isTouching(mouseX,mouseY):
            pygame.draw.rect(screen,self.pressedColor,pygame.Rect(self.x,self.y,self.buttonWidth,self.buttonHeight),0)
            screen.blit(buttonText,(self.x + 10,self.y + int(self.buttonHeight/2)-20))
        else:
            pygame.draw.rect(screen,self.unPressedColor,pygame.Rect(self.x,self.y,self.buttonWidth,self.buttonHeight),0)
            screen.blit(buttonText,(self.x + 10,self.y + int(self.buttonHeight/2)-20))
            
class Pose(object):
    def __init__(self,X,Y):
        self.x = X
        self.y = Y

class Slider(object):
    def __init__(self,X,Y,width,height,maxFrame,frame):
        self.background = (170,170,170)
        self.slider = (50,50,50)
        self.x = X
        self.y = Y
        self.sliderWidth = 30
        self.sliderHeight = height
        self.frame = frame
        self.maxFrame = maxFrame
        self.backWidth = width
        self.backHeight = height/2
    
    def getValue(self):
        return self.frame#/self.maxFrame
    
    def setValue(self,frame):
        self.frame = frame
        
    def slideUpdate(self,mouseX,mouseY):
        if self.isTouching(mouseX,mouseY):
            self.frame = int(((mouseX-self.x)/self.backWidth)*maxFrame)
        
    def isTouching(self,mouseX,mouseY):
        return self.x <= mouseX <= self.x+self.backWidth and self.y <= mouseY <= self.y+self.sliderHeight
    
    def drawSlider(self,screen):
        pygame.draw.rect(screen,self.background,pygame.Rect(self.x,self.y+int(self.backHeight/2),self.backWidth,self.backHeight),0)
        pygame.draw.rect(screen,self.slider,pygame.Rect(self.x+((self.frame/self.maxFrame)*self.backWidth)-int(self.sliderWidth/2),self.y,self.sliderWidth,self.sliderHeight),0)
    
pygame.init()

barHeight = 125


cmd = ['xrandr']
cmd2 = ['grep', '*']
p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
p2 = subprocess.Popen(cmd2, stdin=p.stdout, stdout=subprocess.PIPE)
p.stdout.close()
resolution_string, junk = p2.communicate()
resolution = resolution_string.split()[0]
resolution = resolution.decode("utf-8")
width, height = resolution.split('x')
width = int(int(width)/2)
height = int(int (height)/2)

robot = Pose(0,0)
corners = []

l = 1 #0 for camera in computer and 1 for webcam
vid = cv.VideoCapture(l)
m = 0
while not vid.isOpened():
    m = m + 1
    print('Waiting')
    if (m >= 10):
        m = 0
        l = l + 1
        vid = cv.VideoCapture(l)

vidWidth = vid.get(3)
vidHeight = vid.get(4)
fieldThickness = 5
robotThickness = 10

fieldDim = min(width - vidWidth - fieldThickness, height - barHeight - fieldThickness)

inputData = Button(0,0,vidWidth,barHeight,"Input data from robot")
play =  Button(width - int(fieldDim*1/4),fieldDim+int(fieldThickness/2),int(fieldDim*1/4),barHeight,"Play")
pause = Button(width - int(fieldDim*1/4),fieldDim+int(fieldThickness/2),int(fieldDim*1/4),barHeight,"Pause")
left  = Button(width - fieldDim,        fieldDim+int(fieldThickness/2), int(fieldDim/2), fieldDim, " ")
right = Button(width - int(fieldDim/2), fieldDim+int(fieldThickness/2), int(fieldDim/2), fieldDim, " ")
slider = Slider(width - fieldDim - fieldThickness,fieldDim+int(fieldThickness/2),int(fieldDim*3/4),barHeight,1,0.0)
paused = True

screen = pygame.display.set_mode([width, height],pygame.RESIZABLE)

def getCamFrame(camera):
    retval,frame=camera.read()
    frame=numpy.rot90(frame)
    
    frame=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    
    orig = frame.copy()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (19, 19), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)
    frame = orig.copy()
    return frame

def blitCamFrame(frame,screen):
    screen.blit(frame,(0,barHeight))
    return screen

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return False

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    
    minX = min(min(min(line1[0][0],line1[1][0]),line2[0][0]),line2[1][0])
    maxX = max(max(max(line1[0][0],line1[1][0]),line2[0][0]),line2[1][0])
    minY = min(min(min(line1[0][1],line1[1][1]),line2[0][1]),line2[1][1])
    maxY = max(max(max(line1[0][1],line1[1][1]),line2[0][1]),line2[1][1])
    if (minX <= x <= maxX and minY <= y <= maxY):
        return True
    return False

def drawModel(fieldX,fieldY,fieldDim,x,y,heading):
    y *= -1
    line = fieldDim * 0.0625
    x *= fieldDim/144.0
    y *= fieldDim/144.0
    fieldX += int(fieldDim/2)
    fieldY += int(fieldDim/2)
    pygame.draw.circle(screen, (0, 0, 255), (fieldX + x, fieldY + y), line,robotThickness)
    pygame.draw.line(screen, (0, 0, 255),(fieldX + x,fieldY + y),(line * math.cos(heading) + fieldX + x,-1 * line * math.sin(heading) + fieldY + y),robotThickness)

def drawfield(x,y,width,height,thickness):
    pygame.draw.rect(screen,(200,200,200),pygame.Rect(x,y,width,height),0)
    pygame.draw.rect(screen,0,pygame.Rect(x,y,width,height),thickness)
    for i in range(5):
        pos = int(width/6)*(i+1) + x
        pygame.draw.line(screen,(100,100,100,250),(pos,y),(pos,y+height),thickness)
    for i in range(5):
        pos = int(height/6)*(i+1) + y
        pygame.draw.line(screen,(100,100,100,250),(x,pos),(x+width,pos),thickness)

def matrixFromFourPoints(src, dst):
    if len(src)!=4:
        raise ValueError("src must contain 4 points:", src)
    if len(dst)!=4:
        raise ValueError("dst must contain 4 points:", dst)
    for v in [src,dst]:
        for i in range(4):
            if len(v[i])==2:
                v[i] = [v[i][0],v[i][1],1]
            elif len(v[i])==3:
                if v==[0,0,0]:
                    raise ValueError("no point can be [0,0,0]")
            else:
                raise ValueError("points must have 2 (affine) or 3 (projective) coordinates:", v[i])
    # compute scaling values
    wsrc = [numpy.linalg.det([src[ii] for ii in range(4) if ii!=i]) for i in range(4)]
    wdst = [numpy.linalg.det([dst[ii] for ii in range(4) if ii!=i]) for i in range(4)]
    # If a scaling value is 0, that means 3 points are on the same line (colinear)
    for i in range(4):
        if wsrc[i]==0:
            raise ValueError("3 src points colinear; indices:", [ii for ii in range(4) if ii!=i])
        if wdst[i]==0:
            raise ValueError("3 dst points colinear; indices:", [ii for ii in range(4) if ii!=i])
    # create matrices from first 3 scaled points
    Src = numpy.matrix([[src[c][r]*wsrc[c] for c in range(3)] for r in range(3)])
    Dst = numpy.matrix([[dst[c][r]*wdst[c] for c in range(3)] for r in range(3)])
    # return the transformation matrix
    return numpy.matmul(Dst,numpy.linalg.inv(Src))

def findBlobs(image):
    orig = image.copy()
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    gray = cv.GaussianBlur(gray, (1, 1), 0)
    thresh = cv.threshold(gray, 220, 255, cv.THRESH_BINARY)[1]
    thresh = cv.dilate(thresh, None, iterations=3)
    labels = measure.label(thresh, connectivity=2, background=0)
    mask = numpy.zeros(thresh.shape, dtype="uint8")
    for label in numpy.unique(labels):
            if label == 0:
                    continue
            labelMask = numpy.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv.countNonZero(labelMask)
            if numPixels < 700:
                mask = cv.add(mask, labelMask)
    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if (len(cnts) > 0):
        n = 0
        cnts = contours.sort_contours(cnts)[0]
        
        minR = 10000
        bestC = 0
        
        global robot
        
        for (i,c) in enumerate(cnts):
            (x, y, w, h) = cv.boundingRect(c)
            ((cX, cY), radius) = cv.minEnclosingCircle(c)
            b = pow(pow(cX - robot.y,2) + pow(cY - robot.x,2),0.5)
            if (b < minR):
                minR = b
                bestC = c
        (x, y, w, h) = cv.boundingRect(bestC)
        ((cX, cY), radius) = cv.minEnclosingCircle(bestC)
        pose = Pose(int(cY), int(cX))
        if (minR < 50):
            robot = pose
            cv.circle(image, (int(cX), int(cY)), int(radius),(0, 0, 255), 3)
            cv.putText(image, "#{}".format(5), (x, y - 15),cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
        for a in corners:
            #image[a.x][a.y] = (255,255,255)
            minR = 10000
            bestC = 0
            for (i,c) in enumerate(cnts):
                (x, y, w, h) = cv.boundingRect(c)
                ((cX, cY), radius) = cv.minEnclosingCircle(c)
                b = pow(pow(cX - a.y,2) + pow(cY - a.x,2),0.5)
                image[y][x] = (255,255,255)
                if (b < minR):
                    minR = b
                    bestC = c
            (x, y, w, h) = cv.boundingRect(bestC)
            ((cX, cY), radius) = cv.minEnclosingCircle(bestC)
            pose = Pose(int(cY), int(cX))
            if (minR < 20):
                corners[corners.index(a)] = pose
                cv.circle(image, (int(cX), int(cY)), int(radius),(0, 0, 255), 3)
            n += 1
    if (len(corners) >= 4):
        for i in range(len(corners)):
            cv.line(image,(int(corners[i].y),int(corners[i].x)),(int(corners[(i+1)%len(corners)].y),int(corners[(i+1)%len(corners)].x)),(0, 0, 255), 3)
            cv.putText(image, "#{}".format(i + 1), (int(corners[i].y),int(corners[i].x-15)),cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    frame=pygame.surfarray.make_surface(image)
    return frame
    
    

running = True
inputtedData = False

startTime = time.clock()
data = ("","")
currentRobotFrame = ("","")
mouseDown = False
maxFrame = 1

scrolling = False

leftTime = 0.0
rightTime = 0.0

currentRobotX = 0
currentRobotY = 0

while running:
    
    width = screen.get_width()
    height = screen.get_height()
    
    barHeight = min(int((height)/8),height-vidHeight)
    
    fieldThickness = int((width - vidWidth)/288)
    robotThickness = int((width - vidWidth)/144)
    
    fieldDim = min(width - vidWidth - fieldThickness, height - barHeight - fieldThickness)
    
    inputData = Button(0,0,vidWidth,barHeight,"Input data from robot")
    play =  Button(width - int(fieldDim*1/4),fieldDim+int(fieldThickness/2),int(fieldDim*1/4),barHeight,"Play")
    pause = Button(width - int(fieldDim*1/4),fieldDim+int(fieldThickness/2),int(fieldDim*1/4),barHeight,"Pause")
    left  = Button(width - fieldDim,        0, int(fieldDim/2), fieldDim, " ")
    right = Button(width - int(fieldDim/2), 0, int(fieldDim/2), fieldDim, " ")
    slider = Slider(width - fieldDim - fieldThickness,fieldDim+int(fieldThickness/2),int(fieldDim*3/4),barHeight,maxFrame,slider.getValue())
    
    
    screen.fill((0, 0, 0))
    mouse = pygame.mouse.get_pos()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouseDown = True
            if (mouse[0] < vidWidth and barHeight < mouse[1] < barHeight + vidHeight):
                pose = Pose(mouse[0],mouse[1]-barHeight)
                if (len(corners) >= 4):
                    minR = 10000
                    bestA = 0
                    for a in corners:
                        b = pow(pow(pose.y - a.y,2) + pow(pose.x - a.x,2),0.5)
                        if (b < minR):
                            minR = b
                            bestA = a
                    corners.pop(corners.index(bestA))
                corners.append(pose)
                if (len(corners) >= 4):
                    for i in range(3):
                        minR = 100000
                        bestA = 0
                        q = i % 2
                        u = 0
                        if (i >= 2):
                            q = 1
                            u = 1
                        target = Pose(q * 640, u * 480)
                        for j in range(4-i):
                            b = pow(pow(target.y - corners[3-j].y,2) + pow(target.x - corners[3-j].x,2),0.5)
                            if (b < minR):
                                minR = b
                                bestA = 3-j
                        (corners[i+1], corners[bestA]) = (corners[bestA], corners[i+1])
                    if (line_intersection(((corners[0].x,corners[0].y),(corners[1].x,corners[1].y)),((corners[2].x,corners[2].y),(corners[3].x,corners[3].y)))):
                        print("here 1")
                        (corners[0], corners[3]) = (corners[3], corners[0])
                    if (line_intersection(((corners[0].x,corners[0].y),(corners[3].x,corners[3].y)),((corners[2].x,corners[2].y),(corners[1].x,corners[1].y)))):
                        print("here 2")
                        (corners[2], corners[3]) = (corners[3], corners[2])
                        
            if inputData.isTouching(mouse[0],mouse[1]):
                TextEditorTest.getRobotLog()
                data = refineData.refineData()
                maxFrame = len(data)
                inputtedData = True
            if inputtedData:
                if play.isTouching(mouse[0],mouse[1]):
                    paused = not paused
                    if not paused and currentFrame == maxFrame -1:
                        slider.setValue(0)
                if left.isTouching(mouse[0],mouse[1]):
                    if time.clock()-leftTime < 0.5:
                        leftTime = 0
                        slider.setValue(max(slider.getValue() - 120,0))
                        print("left")
                    leftTime = time.clock()
                if right.isTouching(mouse[0],mouse[1]):
                    if time.clock()-rightTime < 0.5:
                        rightTime = 0
                        slider.setValue(min(slider.getValue() + 120,maxFrame-1))
                        print("right")
                    rightTime = time.clock()
        if event.type == pygame.MOUSEBUTTONUP:
            mouseDown = False
        if event.type == pygame.KEYDOWN:
            print(event.key)
            if event.key == 32:
                robot = Pose(mouse[0],mouse[1]-barHeight)
            if event.key == 1073741903 and inputtedData and currentFrame < maxFrame-1:
                slider.setValue(slider.getValue() + 1)
            if event.key == 1073741904 and inputtedData and currentFrame > 0:
                slider.setValue(slider.getValue() - 1)
            
    inputData.drawButton(screen,mouse[0],mouse[1])
    
    if (len(corners) == 4):
        M = matrixFromFourPoints([[corners[0].x,corners[0].y],[corners[1].x,corners[1].y],[corners[2].x,corners[2].y],[corners[3].x,corners[3].y]], [[0,0],[1,0],[1,1],[0,1]])
        w = numpy.matmul(M, numpy.transpose([[robot.x,robot.y,1]])) # v is robot possition in form (x,y,1)
        w /= w[2,0]
        currentRobotX = (w[0][0]-0.5) * 72 * 2
        currentRobotY = (w[1][0]-0.5) * 72 * 2
    
    slider.drawSlider(screen)
    scrolling = False
    if mouseDown:
        slider.slideUpdate(mouse[0],mouse[1])
        if slider.isTouching(mouse[0],mouse[1]):
            scrolling = True
    
    if paused or scrolling:
        startTime = time.clock();
        play.drawButton(screen,mouse[0],mouse[1])
    else:
        while time.clock() - startTime > 1.0/24.0:
            startTime += 1.0/24
            slider.setValue(slider.getValue() + 1)
        pause.drawButton(screen,mouse[0],mouse[1])
    
    frame = findBlobs(getCamFrame(vid))
    frame = blitCamFrame(frame,screen)
    drawfield(width - fieldDim - int(fieldThickness/2),0,fieldDim,fieldDim,fieldThickness)
    
    currentFrame = int(slider.getValue())
    if inputtedData:
        if currentFrame >= len(data)-1:
            vidTime = len(data)-1
            paused = True
        else:
            currentRobotFrame = data[currentFrame].split(",",2)
        if len(currentRobotFrame)==3:
            drawModel(width - fieldDim - int(fieldThickness/2), 0, fieldDim, float(currentRobotFrame[0]),float(currentRobotFrame[1]),float(currentRobotFrame[2]))
    #else:
        #drawModel(width - fieldDim - int(fieldThickness/2),0,fieldDim,0,0,0)
    
    drawModel(width - fieldDim - int(fieldThickness/2),0,fieldDim,currentRobotX,currentRobotY,0)
    
    pygame.display.flip()

pygame.quit()
vid.release()

