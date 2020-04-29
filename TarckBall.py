from pyrr import Matrix44,Vector3
import pyrr
import math
#http://web.cse.ohio-state.edu/~shen.94/781/Site/Slides_files/trackball.pdf#
class TrackBall():
    #x means window width, y means window height,d means the size of sphere. 
    def __init__(self,x,y,radius): #
        self.windowWidth=x
        self.windowHeight=y
        self.radius=radius
        self.startX=0
        self.startY=0
        self.trackballMove=False
        self.curPos=Vector3([0,0,0])
        self.lastPos=Vector3([0,0,0])
        self.angle=0
        self.axis=Vector3([1,0,1])
        self.doingAnything=True
    def pTov(self,x,y,radius):
        x = (2.0 * x) /self.windowWidth - 1.0
        y = 1.0 - (2.0 * y) /self.windowHeight
        d=math.sqrt(x*x+y*y)
        z=0
        if d<radius:
            z=math.sqrt(radius*radius-d*d)
            return Vector3([x,y,z])
        else:
            z=0
            return Vector3([x/d*radius,y/d*radius,z])
    def startMotion(self,x,y):
        #called when mouse is pressed
        self.startX=x
        self.startY=y
        #print("start x",self.startX,self.startY)
        self.trackballMove=True
        #print("radius is",self.radius)
        self.lastPos=self.pTov(x,y,1)
        
    def mouseMotion(self,x,y):    #called continously when it is pressed
        self.curPos=self.pTov(x,y,1)
        changePos=self.curPos-self.lastPos
        self.notdoingAnything=False
        n=self.curPos^self.lastPos
        self.angle=90*pyrr.vector.length(changePos)
        self.lastPos=self.curPos
        self.axis=n
        #print(self.angle)
    def getRotationMatrix(self):    # newRotate=NewRotate*oldrotate
            if not self.angle:
                return Matrix44.identity()
            temp=pyrr.matrix44.create_from_axis_rotation(self.axis,math.radians(self.angle))
            return Matrix44(temp)
    def isTrackBallMove(self):  #ask if trackball first before apply rotation matrix
        return self.trackballMove
    def mouseStopMotion(self):
        self.trackballMove=False

'''
mouseEventClick(x,y)
if Trackball.trackballMove() is false:
    start
    mouseMotion
else:
    mouseMotion

mousereleaseEvent(x,y)



update()
is trackballMove then apply getRotationMatrix
else dont apply 
'''


