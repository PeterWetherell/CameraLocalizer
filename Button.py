class Button(object):
    pressedColor = (170,170,170)
    unPressedColor = (100,100,100)
    
    def __init__(self,X,Y,width,height,displayText):
        self.x = X
        self.y = Y
        self.buttonWidth = width
        self.buttonHeight = height
        self.text = displayText
        
    def isTouching(self,mouseX,mouseY):
        return self.x <= mouseX <= self.x+self.buttonWidth and self.y <= mouseY <= self.y+self.buttonHeight