#https:#stackoverflow.com/questions/7915201/opengl-rubiks-cube-face-rotation-with-mouse
import numpy as np
import time
import math
from qtmoderngl import QModernGLWidget
import sys
import pyrr 
from pyrr import Matrix44,Vector3,Vector4,Quaternion
import PyQt5

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget,QPushButton,QLabel,QHBoxLayout,QVBoxLayout,QMainWindow,QSizePolicy

from renderer_example import HelloWorld2D, PanTool
import moderngl
import moderngl_window
from TrackBall import TrackBall


#adjust the speed here, 1 means a bit slow, 
SPEED=6


#face type
FRONT=2
TOP=1
BOTTOM=3
LEFT=0
RIGHT=4
BACK=5
NONE=-1
#face color

RED=[1.0,0.0,0.0]
BLUE=[0.0,0.0,1.0]
GREEN=[0.0,1.0,0.0]
WHITE=[1.0,1.0,1.0]
YELLOW=[1.0,1.0,0.0]
ORANGE=[1.0,0.5,0.5]
BLACK=[0.0,0.0,0.0]

CUBEFACE=[[0,2,3],[2,3],[2,3,4],[0,2],[2],[2,4],[0,1,2],[1,2],[1,2,4],[0,3],[3],[4,3],[0],[],[4],[0,1],[1],[1,4],[0,5,3],[5,3],[3,4,5],[0,5],[5],[5,4],[0,1,5],[1,5],[1,4,5]]

#CUBEFACE=np.asarray([np.pad(a, (0, 6 - len(a)), 'constant', constant_values=NONE) for a in CUBEFACE_temp])

#index 0 and 5 are the same, index, 
#index 2 and 3 are the same 


CLOCKWISE=1
ANTICLOCKWISE=-1


VERTICES= np.array([
            [[-0.5, -0.5, -0.5],               
    [0.5, -0.5, -0.5],
    [0.5,  0.5, -0.5],
    [0.5,  0.5, -0.5],
    [-0.5,  0.5, -0.5],
    [-0.5, -0.5, -0.5]],# one face faceid 2
    
    [[-0.5, -0.5,  0.5],#6
    [0.5, -0.5,  0.5],
    [0.5,  0.5,  0.5],
    [0.5,  0.5,  0.5],
    [-0.5,  0.5,  0.5],
    [-0.5, -0.5,  0.5]],#two face   faceid 5
    
    [[-0.5,  0.5,  0.5],#12
    [-0.5,  0.5, -0.5],
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5,  0.5],
    [-0.5,  0.5,  0.5]],#three face  faceid4
    
    [[0.5,  0.5,  0.5],#18
    [0.5,  0.5, -0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5,  0.5],
    [0.5,  0.5,  0.5]],#four face   faceid0
    
    [[-0.5, -0.5, -0.5],#24
    [0.5, -0.5, -0.5],
    [0.5, -0.5,  0.5],
    [0.5, -0.5,  0.5],
    [-0.5, -0.5,  0.5],
    [-0.5, -0.5, -0.5]],# five face faceid 3
    
    [[-0.5,  0.5, -0.5],
    [0.5,  0.5, -0.5],
    [0.5,  0.5,  0.5],
    [0.5,  0.5,  0.5],
    [-0.5,  0.5,  0.5],
    [-0.5,  0.5, -0.5]],# six face faceid 1
],dtype='f4')


class mainScreen(QWidget):
    def __init__(self, parent=None):
        super(mainScreen, self).__init__(parent)
        
        #To Make Background Fit the Window
        self.shuffleBTN=QPushButton("SHUFFLE",self)
        self.shuffleBTN.clicked.connect(self.test)    #connect the fucntion name, not the fucniton itself
        #self.lbl2 = QLabel('OPENGL', self)
        #self.lbl2.move(35, 40)
        self.widget =MyWidget()
        vbox = QHBoxLayout()
        vbox.setSpacing(0)
        #vbox.addStretch(2)
        side= QVBoxLayout()
        side.setSpacing(0)
        
        #vbox.addWidget(self.lbl2)
        #vbox.addStretch(2)
        vbox.setContentsMargins(0,0,0,0)
        #vbox.addStretch(4)
        #self.widget.setSizePolicy(left)
        vbox.addWidget(self.widget,3)
        #vbox.addStretch(2)
        #self.shuffleBTN.setSizePolicy(right)
        side.addWidget(self.shuffleBTN,1)
        vbox.addLayout(side)
        self.setLayout(vbox)
        self.setGeometry(300, 300, 900, 700)
        self.setWindowTitle('AI solving rubic cube')
        self.setMinimumSize(900, 700)
        self.show()

        
    #https://doc.qt.io/qt-5/qt.html
    def keyPressEvent(self, evt):
        
        if evt.key()==Qt.Key_Escape:
            self.close()
        if evt.key()==Qt.Key_Space:
            self.widget.scene.test()
        if evt.key()==Qt.Key_A:
            self.widget.scene.test2()
    def test(self):
        self.widget.scene.rotateRandomly(1)



    




class MyWidget(QModernGLWidget):
    def __init__(self):
        super(MyWidget, self).__init__()
        self.scene = None

        self._start_time=time.time()
        self.time=0
        self.LeftButtonDown=False
        self.rightButtonDown=False
        #self.setMouseTracking(True)

    def init(self):
        self.ctx.viewport = (0, 0, self.width(),self.height())
        #print(self.width()) 801
        #print(self.height()) 700
        self.scene =RubikCube(self.ctx,self.width(),self.height()) #RubikCube
        #self.resize(100, 500)
        
        #self.scene.setTranslation(4,9,1)
        #self.scene=face(self.ctx,1)

        self.loop=PyQt5.QtCore.QTimer()
        self.loop.start(10)
        self.loop.timeout.connect(self.update) # minstake here, dont use(), 
        #if u use(),u are executing the function

    def shuffle(self):
        self.scene.rotateRandomly(1)
    def render(self):
        #self.scene.updateParameter()
        self.scene.render()
        self.time=time.time()-self._start_time
    def mousePressEvent(self, evt):
        if evt.button()==Qt.LeftButton:
            #right clicked
            #self.scene.test()
            #self.scene.RightClicked(evt.x(),evt.y())
            self.scene.LeftClicked(evt.x(),evt.y())
            self.LeftButtonDown=True
            
        #print(evt.x())
        elif evt.button()==Qt.RightButton:
            self.rightButtonDown=True
            self.scene.RightClicked(evt.x(),evt.y())
    def mouseMoveEvent(self,evt):
        #print(evt.x())
        #left drag and right drag must be separate 
        if self.LeftButtonDown and self.rightButtonDown:
            print("dont press two button at the same time, u will crash the program")
        elif self.LeftButtonDown:
            self.scene.LeftDrag(evt.x(),evt.y())
        elif self.rightButtonDown:
            self.scene.RightDrag(evt.x(),evt.y())
        
    def mouseReleaseEvent(self, evt):
        #pan_tool.stop_drag(evt.x() / 512, evt.y() / 512)
        self.scene.release()
        self.LeftButtonDown=False
        self.rightButtonDown=False
    def keyPressEvent(self, evt):
        print("hi")
        if evt.key()==Qt.Key_Escape:
            print("escape")
        print(evt.key())


#ctx is moderngl context. 


class Face:
    def __init__(self, ctx,width,height,Type,id):
        self.ctx = ctx
        self.type=Type
        self.ParentCube=id
        self.isImportant=False
        self.color=BLACK
        self.width=width
        self.height=height
        self.x_rot = 0
        self.y_rot = 0
        self.x_last=0
        self.y_last=0
        self.x_cur=0
        self.y_cur=0
        self.isRotating=False
        self.orientation=pyrr.quaternion.create(0.0,0.0,0.0,1.0)
        self.proj = Matrix44.perspective_projection(45.0, self.width/self.height, 0.1, 100.0)
        self.camera_pos = [0,3,-8.0]
        self.vertexes=VERTICES[self.fromTypeTOIndex(Type)]
        self.rotation=Matrix44.identity()
        self.rotationLayer=Matrix44.identity()
        self.degree=0
        self.translation=Matrix44.identity()
        self.view = Matrix44.look_at(
            self.camera_pos, # position of camera in world coordinate
            (0.0, 0.0, 0.0), # target
            (0.0, 1.0, 0.0),
        )
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330

                uniform mat4 Mvp;
                in vec3 in_vert;
                in vec3 in_color;
                out vec3 v_color;  
                out vec4 Position;
                void main() {
                    gl_Position = Mvp*vec4(in_vert, 1.0);
                    v_color = in_color;
                    Position=vec4(in_vert,1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                in vec3 v_color;
                in vec4 Position;
                out vec4 f_color;
                void main() 
                {
                    f_color=vec4(0,0,0,0.9);
                    if (Position.z==-0.5 || Position.z==0.5)
                    {
                        if(Position.x > -0.48 && Position.x < 0.48 && Position.y > -0.48 && Position.y < 0.48)
                            {
                                f_color = vec4(v_color,1.0);
                            }
                    }
                    if (Position.x==-0.5 || Position.x==0.5)
                    {
                        if(Position.z > -0.48 && Position.z < 0.48 && Position.y > -0.48 && Position.y < 0.48)
                            {
                                f_color = vec4(v_color,1.0);
                            }
                    }
                    if (Position.y==-0.5 || Position.y==0.5)
                    {
                        
                        if(Position.z > -0.48 && Position.z < 0.48 && Position.x > -0.48 && Position.x < 0.48)
                            {
                                f_color = vec4(v_color,1.0);
                            }
                    }
                    
                }

            ''',
        )
    def setUp(self):
        self.mvp = self.prog['Mvp']
        vertices =self.generateVertexes(self.type,self.color)
        #self.vbo = self.ctx.buffer(vertices.astype('f4').tobytes())
        self.vbo = self.ctx.buffer(vertices)
        #vbo just means memory. 



        #self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert')
        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                # Map in_vert to the first 3 floats
                # Map in_color to the next 3 floats
                (self.vbo, '3f 3f', 'in_vert', 'in_color')
            ],
        )

    def pan(self, pos):
        self.prog['Pan'].value = pos
    def clicked(self,x,y):
        self.x_cur=x
        self.y_cur=y
    def drag(self,x,y):
        self.x_cur=x
        self.y_cur=y
        self.x_rot -= (self.x_cur-self.x_last) / 1000
        self.y_rot -= (self.y_cur-self.y_last) / 1000


    def clear(self, color=(0, 0, 0, 0)):
        self.ctx.clear(*color)
    def setTranslation(self,temp_matrix):
        self.translation=temp_matrix
    def applyRotation(self,temp_matrix):
        self.rotation=temp_matrix*self.rotation #it is accumulative
    def render(self):
        #self.ctx.clear(0.0,0.0,0.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        #self.model=self.rotation*self.translation
        self.update()
        self.model=self.rotation*self.rotationLayer*self.translation
        self.mvp.write((self.proj * self.view*self.model).astype('f4').tobytes())
        self.vao.render()

    def generateVertexes(self,n,color):
        b=np.array(color,dtype='f4')
        a=self.vertexes
        b_new = np.broadcast_to(b,(a.shape[0],b.shape[0]))
        c = np.concatenate((a,b_new),axis=1)
        return c.flatten()
    def setImportance(self):
        self.isImportant=True
        self.color=self.fromTypeToColor(self.type)

    def update(self):
        if(self.isRotating):
            temp=pyrr.quaternion.create_from_axis_rotation(Vector3(self.direction),math.radians(self.degree))
            aroundX=pyrr.quaternion.cross(self.orientation,temp)  # the order matters a lot.  temp,orientation is wrong. 
            self.rotationLayer=Matrix44(aroundX)
            self.degree+=SPEED
            if((self.degree-90)>0.01):
                self.isRotating=False
                self.orientation=pyrr.quaternion.cross(self.orientation,pyrr.quaternion.create_from_axis_rotation(Vector3(self.direction),math.radians(90)))
                self.degree=0
    def rotateFront(self,direction_):
        if self.isRotating==False:
            self.isRotating=True
            self.direction=[0.0,0.0,-1.0*direction_]
    def rotateLeft(self,direction_):
        if self.isRotating==False:
            self.isRotating=True
            self.direction=[1.0*direction_,0.0,0.0]
    def rotateUp(self,direction_):
        if self.isRotating==False:
            self.isRotating=True
            self.direction=[0.0,1.0*direction_,0.0]
            #print(self.direction)
    
    
    
    @staticmethod
    def fromTypeToColor(n):
        if n==FRONT:
            return GREEN
        if n==BACK:
            return BLUE        
        if n==LEFT:
            return RED
        if n==RIGHT:
            return ORANGE
        if n==TOP:
            return YELLOW
        if n==BOTTOM:
            return WHITE
    @staticmethod
    def fromTypeTOIndex(n):
        if n==2:
            return 0
        if n==5:
            return 1
        if n==4:
            return 2
        if n==0:
            return 3
        if n==3:
            return 4
        if n==1:
            return 5


class FaceType2:
    def __init__(self, ctx,width,height,Type,id):
        self.ctx = ctx
        self.type=Type
        self.ParentCube=id
        self.color=self.fromTypeToColor(self.type)
        self.width=width
        self.height=height
        self.x_rot = 0
        self.y_rot = 0
        self.x_last=0
        self.y_last=0
        self.x_cur=0
        self.y_cur=0
        self.isRotating=False
        self.orientation=pyrr.quaternion.create(0.0,0.0,0.0,1.0)
        self.proj = Matrix44.perspective_projection(45.0, self.width/self.height, 0.1, 100.0)
        self.camera_pos = [0,3,-8.0]
        self.vertexes=VERTICES[self.fromTypeTOIndex(Type)]
        self.rotation=Matrix44.identity()
        self.rotationLayer=Matrix44.identity()
        self.degree=0
        self.translation=Matrix44.identity()
        self.view = Matrix44.look_at(
            self.camera_pos, # position of camera in world coordinate
            (0.0, 0.0, 0.0), # target
            (0.0, 1.0, 0.0),
        )
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330

                uniform mat4 Mvp;
                in vec3 in_vert;
                in vec3 in_color;
                out vec3 v_color;  
                out vec4 Position;
                void main() {
                    gl_Position = Mvp*vec4(in_vert, 1.0);
                    v_color = in_color;
                    Position=vec4(in_vert,1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                in vec3 v_color;
                in vec4 Position;
                out vec4 f_color;
                
                void main() 
                {
                    f_color=vec4(v_color,1);
                   
                    
                }

            ''',
        )
    def setUp(self):
        self.mvp = self.prog['Mvp']
        vertices =self.generateVertexes(self.type,self.color)
       #self.vbo = self.ctx.buffer(vertices.astype('f4').tobytes())
        self.vbo = self.ctx.buffer(vertices)
        #vbo just means memory. 



        #self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert')
        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                # Map in_vert to the first 3 floats
                # Map in_color to the next 3 floats
                (self.vbo, '3f 3f', 'in_vert', 'in_color')
            ],
        )

    def pan(self, pos):
        self.prog['Pan'].value = pos
    def clicked(self,x,y):
        #self.x_last=self.x_cur=x
        #self.y_last=self.y_cur=y
        self.x_cur=x
        self.y_cur=y
        self.MouseRay=self.calculateMouseRay(self.proj,self.view,self.width,self.height,self.x_cur,self.y_cur)
    def drag(self,x,y):
        self.x_cur=x
        self.y_cur=y
        self.x_rot -= (self.x_cur-self.x_last) / 1000
        self.y_rot -= (self.y_cur-self.y_last) / 1000


    def clear(self, color=(0, 0, 0, 0)):
        self.ctx.clear(*color)
    def setTranslation(self,temp_matrix):
        self.translation=temp_matrix
    def applyRotation(self,temp_matrix):
        self.rotation=temp_matrix*self.rotation #it is accumulative
        #print("applying",self.rotation)

    #def updateParameter(self):
        #translation
        #self.rotation = Matrix44.from_eulers((self.y_rot, self.x_rot, 0), dtype='f4')
        
        
        
        #print(pointerCoordinate)
    def render(self):
        #self.ctx.clear(0.0,0.0,0.0)
        #self.ctx.enable(moderngl.DEPTH_TEST)
        #self.model=self.rotation*self.translation
        self.update()
        self.model=self.rotation*self.rotationLayer*self.translation
        # we dont render face type 2 it is only for selecting the face type 
        self.mvp.write((self.proj * self.view*self.model).astype('f4').tobytes())
        self.vao.render()


    def calculateMouseRay(self,projectionMatrix,viewMAtrix,screenX,screenY,mouse_x,mouse_y):
        x = (2.0 * mouse_x) /screenX - 1.0
        y = 1.0 - (2.0 * mouse_y) /screenY
        z = -1.0
        D_view= Vector4.from_vector3(Vector3([x,y,z]), w=1.0)
        ray_eye=projectionMatrix.inverse*D_view

        ray_eye=Vector4([ray_eye.x,ray_eye.y,-1.0, 0.0])

        ray_wor = ((viewMAtrix.inverse) * ray_eye).xyz
        return Vector3(ray_wor)
    def distanceToMouse(self):
        #https://stackoverflow.com/questions/45893277/is-it-possible-get-which-surface-of-cube-will-be-click-in-opengl
        P0=self.model*Vector3(self.vertexes[0])
        P1=self.model*Vector3(self.vertexes[1])
        P2=self.model*Vector3(self.vertexes[2])

        NV=(P1-P0)^(P2-P0)
        distance=(pyrr.vector.dot(P0-Vector3(self.camera_pos), NV)/ pyrr.vector.dot(self.MouseRay,NV))
        return distance
    def isInside(self):
        A=self.model*Vector3(self.vertexes[0])  #ABC ACD 
        B=self.model*Vector3(self.vertexes[1])
        C=self.model*Vector3(self.vertexes[2])
        D=self.model*Vector3(self.vertexes[4])
        def PointINorOn(P1,P2,A1,A2):
            CP1=(A2-A1)^(P1-A1)
            CP2=(A2-A1)^(P2-A1)
            return (CP1|CP2)>=0
        def PointInOrOnTriangle(P,A,B,C):
            return PointINorOn(P,A,B,C) and PointINorOn(P,B,C,A) and PointINorOn(P,C,A,B)
    
        P=self.distanceToMouse()*self.MouseRay+Vector3(self.camera_pos)
        return PointInOrOnTriangle(P,A,B,C) or PointInOrOnTriangle(P,A,C,D)

    def generateVertexes(self,n,color):
        b=np.array(color,dtype='f4')
        a=self.vertexes
        b_new = np.broadcast_to(b,(a.shape[0],b.shape[0]))

        c = np.concatenate((a,b_new),axis=1)
        return c.flatten()
    def setImportance(self):
        self.isImportant=True
        self.color=self.fromTypeToColor(self.type)
        #print("hi")
    def update(self):
        if(self.isRotating):
            temp=pyrr.quaternion.create_from_axis_rotation(Vector3(self.direction),math.radians(self.degree))
            aroundX=pyrr.quaternion.cross(self.orientation,temp)  # the order matters a lot.  temp,orientation is wrong. 
            self.rotationLayer=Matrix44(aroundX)
            self.degree+=1
            if(self.degree==91):
                self.isRotating=False
                self.orientation=aroundX
                self.degree=0
    def rotateFront(self,direction_):
        if self.isRotating==False:
            self.isRotating=True
            self.direction=[0.0,0.0,-1.0*direction_]
    def rotateLeft(self,direction_):
        if self.isRotating==False:
            self.isRotating=True
            self.direction=[1.0*direction_,0.0,0.0]
    def rotateUp(self,direction_):
        if self.isRotating==False:
            self.isRotating=True
            self.direction=[0.0,1.0*direction_,0.0]
            #print(self.direction)
    
    
    
    @staticmethod
    def fromTypeToColor(n):
        if n==FRONT:
            return GREEN
        if n==BACK:
            return BLUE        
        if n==LEFT:
            return RED
        if n==RIGHT:
            return ORANGE
        if n==TOP:
            return YELLOW
        if n==BOTTOM:
            return WHITE
    @staticmethod
    def fromTypeTOIndex(n):
        if n==2:
            return 0
        if n==5:
            return 1
        if n==4:
            return 2
        if n==0:
            return 3
        if n==3:
            return 4
        if n==1:
            return 5



        
class Cube:
    def __init__(self,ctx,width,height,id):
        self.ctx=ctx
        self.faces=[Face(ctx,width,height,i,id) for i in range(6)]
        self.ID=id
        for i in CUBEFACE[id]:
            self.faces[i].setImportance()
        for i in self.faces:
            i.setUp()
        self.translation=Matrix44.identity()
        self.rotation=Matrix44.identity()

    def render(self):
        #self.ctx.clear(0.0,0.0,0.0)
        for i in self.faces:
            #i.updateParameter()
            i.render()

    def setTranslation(self,up,right,behind):  #only done once by the Cube. 
        #up=Vector3([0,up,0])
        #right=Vector3([right,0,0])
        #behind=Vector3([0,0,-1*behind])
        self.translation=Matrix44.from_translation([-right*1.0+1.0,up*1.0-1.0,1.0*behind-1.0])
        #print(self.translation)
        for i in self.faces:
            i.setTranslation(self.translation)
    def applyRotation(self,globalRotation):
        for i in self.faces:
            i.applyRotation(globalRotation)
    def printColor(self):
        for i in self.faces:
            print(i.color)
    def rotateFront(self,direction_):
        for i in self.faces:
            i.rotateFront(direction_)
    def rotateLeft(self,direction_):
        for i in self.faces:
            i.rotateLeft(direction_)
    def rotateUp(self,direction_):
        for i in self.faces:
            i.rotateUp(direction_)
    def isRotating(self):
        for i in self.faces:
            if i.isRotating:
                return True
        return False


class CubeType2:
    def __init__(self,ctx,width,height,id):
        self.ctx=ctx
        self.faces=[FaceType2(ctx,width,height,i,id) for i in CUBEFACE[id]]
        self.ID=id
        for i in self.faces:
            #print("setting up")
            i.setUp()
        self.translation=Matrix44.identity()
        self.rotation=Matrix44.identity()

    def render(self):
        #self.ctx.clear(0.0,0.0,0.0)
        for i in self.faces:
            #i.updateParameter()
            i.render()
    def clicked(self,x,y): # return -1 if no face is clicked. 
        mindistance=1000
        id=-1
        for i in self.faces:
            i.clicked(x,y)
        for i in self.faces:
            temp=i.distanceToMouse()
            #print(self.ID,"distance is",temp)
            if i.isInside() and temp<mindistance:
                mindistance=temp
                id=i.type
        #print(id)
        return (id,mindistance)

    def setTranslation(self,up,right,behind):  #only done once by the Cube. 
        #up=Vector3([0,up,0])
        #right=Vector3([right,0,0])
        #behind=Vector3([0,0,-1*behind])
        self.translation=Matrix44.from_translation([-right*1.0+1.0,up*1.0-1.0,1.0*behind-1.0])
        #print(self.translation)
        for i in self.faces:
            i.setTranslation(self.translation)
    def applyRotation(self,globalRotation):
        for i in self.faces:
            i.applyRotation(globalRotation)





            



class RubikCube():
    def __init__(self,ctx,width,height):
        self.trackball=TrackBall(width,height,1)
        self.ctx=ctx
        self.cubes=[Cube(ctx,width,height,i) for i in range(27)]
        self.cubesType2=[CubeType2(ctx,width,height,i) for i in range(27)]
        self.cubeMove=np.arange(27).reshape((3, 3,3))   # it keeps track of which index is at which location of the cube. 
        for i in range(27):
            self.cubes[i].setTranslation((int)((i%9)/3),(i%9)%3,(int)(i/9))
            self.cubesType2[i].setTranslation((int)((i%9)/3),(i%9)%3,(int)(i/9))

        #self.cubes[0].printColor()
        self.Prev_Right=(-1,-1)         #store the current right clicked cube
        self.receivingDragFroce=True
            

    def render(self):
       

        #if self.trackball.isTrackBallMove():
            #print(self.trackball.doingAnything)
            #print(self.trackball.getRotationMatrix())

            #
                #
        self.ctx.clear(0.0,0.7,0.7)
        self.ctx.enable(moderngl.BLEND)
        for i in self.cubesType2:
            #print("cube")
            i.render()
        for i in self.cubes:
            i.render()
    def LeftClicked(self,x,y):
        self.trackball.startMotion(x,y)
        #print("hi")
    def RightClicked(self,x,y):
        self.Prev_Right=self.RightClicked_help(x,y)
    def RightClicked_help(self,x,y):
        #cube id=-1,face Type=-1  minimum distance=1000
        id=-1
        faceType=-1
        minDistance=100
        for i in self.cubesType2:
            result=i.clicked(x,y) # result=[faceType,distance]
            if result[0]!=-1:
                if result[1]<minDistance:
                    id=i.ID
                    faceType=result[0]
                    minDistance=result[1]
        #print(id,faceType)
        return(id,faceType)

        #get faceType and cube id. 
    def RightDrag(self,x,y):
        if self.receivingDragFroce and not self.isRotating() :
            temp=self.RightClicked_help(x,y)
            if temp[0]!=-1:
                if self.Prev_Right[0]!=temp[0]:
                    self.receivingDragFroce=False
                    print("doing drag force")
                    print(self.Prev_Right)
                    print(temp)
                    self.doRotation(temp[1],self.Prev_Right[0],temp[0])
                    self.receivingDragFroce=True
                    self.Prev_Right=temp
    def doRotation(self,_facetype,_cube1,_cube2):
        if _facetype==FRONT:
            if (_cube1==0 and _cube2==1) or (_cube1==1 and _cube2==2):
                 self.UpRotate(2,ANTICLOCKWISE)
            elif (_cube1==3 and _cube2==4) or (_cube1==4 and _cube2==5):
                 self.UpRotate(1,ANTICLOCKWISE)
            elif (_cube1==6 and _cube2==7) or (_cube1==7 and _cube2==8):
                 self.UpRotate(0,ANTICLOCKWISE)
            elif (_cube1==1 and _cube2==0) or (_cube1==2 and _cube2==1):
                 self.UpRotate(2,CLOCKWISE)
            elif (_cube1==4 and _cube2==3) or (_cube1==5 and _cube2==4):
                 self.UpRotate(1,CLOCKWISE)
            elif (_cube1==7 and _cube2==6) or (_cube1==8 and _cube2==7):
                 self.UpRotate(0,CLOCKWISE)
            ##
            elif (_cube1==0 and _cube2==3) or (_cube1==3 and _cube2==6):
                self.LeftRotate(0,ANTICLOCKWISE)
            elif (_cube1==1 and _cube2==4) or (_cube1==4 and _cube2==7):
                self.LeftRotate(1,ANTICLOCKWISE)    
            elif (_cube1==2 and _cube2==5) or (_cube1==5 and _cube2==8):
                self.LeftRotate(2,ANTICLOCKWISE) 
            elif (_cube1==3 and _cube2==0) or (_cube1==6 and _cube2==3):
                self.LeftRotate(0,CLOCKWISE)
            elif (_cube1==2 and _cube2==1) or (_cube1==7 and _cube2==4):
                self.LeftRotate(1,CLOCKWISE)    
            elif (_cube1==5 and _cube2==2) or (_cube1==8 and _cube2==5):
                self.LeftRotate(2,CLOCKWISE)
            else:
                print("that is impossible")
        if _facetype==LEFT:
            if (_cube1==18 and _cube2==9) or (_cube1==9 and _cube2==0):
                 self.UpRotate(2,ANTICLOCKWISE)
            elif (_cube1==21 and _cube2==12) or (_cube1==12 and _cube2==3):
                 self.UpRotate(1,ANTICLOCKWISE)
            elif (_cube1==24 and _cube2==15) or (_cube1==15 and _cube2==6):
                 self.UpRotate(0,ANTICLOCKWISE)
            elif (_cube1==9 and _cube2==18) or (_cube1==0 and _cube2==9):
                 self.UpRotate(2,CLOCKWISE)
            elif (_cube1==12 and _cube2==21) or (_cube1==3 and _cube2==12):
                 self.UpRotate(1,CLOCKWISE)
            elif (_cube1==15 and _cube2==24) or (_cube1==6 and _cube2==15):
                 self.UpRotate(0,CLOCKWISE)
            #
            elif (_cube1==24 and _cube2==21) or (_cube1==21 and _cube2==18):
                 self.FrontRotate(2,ANTICLOCKWISE)
            elif (_cube1==15 and _cube2==12) or (_cube1==12 and _cube2==9):
                 self.FrontRotate(1,ANTICLOCKWISE)
            elif (_cube1==6 and _cube2==3) or (_cube1==3 and _cube2==0):
                 self.FrontRotate(0,ANTICLOCKWISE)
            elif (_cube1==21 and _cube2==24) or (_cube1==18 and _cube2==21):
                 self.FrontRotate(2,CLOCKWISE)
            elif (_cube1==12 and _cube2==15) or (_cube1==9 and _cube2==12):
                 self.FrontRotate(1,CLOCKWISE)
            elif (_cube1==3 and _cube2==6) or (_cube1==0 and _cube2==3):
                 self.FrontRotate(0,CLOCKWISE)
            else:
                print("it is impossible")
        if _facetype==TOP:
            if (_cube1==6 and _cube2==7) or (_cube1==7 and _cube2==8):
                 self.FrontRotate(0,CLOCKWISE)
            elif (_cube1==15 and _cube2==16) or (_cube1==16 and _cube2==17):
                 self.FrontRotate(1,CLOCKWISE)
            elif (_cube1==24 and _cube2==25) or (_cube1==25 and _cube2==26):
                 self.FrontRotate(2,CLOCKWISE)
            elif (_cube1==7 and _cube2==6) or (_cube1==8 and _cube2==7):
                 self.FrontRotate(0,ANTICLOCKWISE)
            elif (_cube1==16 and _cube2==15) or (_cube1==17 and _cube2==16):
                 self.FrontRotate(1,ANTICLOCKWISE)
            elif (_cube1==25 and _cube2==24) or (_cube1==26 and _cube2==25):
                 self.FrontRotate(2,ANTICLOCKWISE)
            #
            elif (_cube1==24 and _cube2==15) or (_cube1==15 and _cube2==6):
                 self.LeftRotate(0,CLOCKWISE)
            elif (_cube1==25 and _cube2==16) or (_cube1==16 and _cube2==7):
                 self.LeftRotate(1,CLOCKWISE)
            elif (_cube1==26 and _cube2==17) or (_cube1==17 and _cube2==8):
                 self.LeftRotate(2,CLOCKWISE)
            elif (_cube1==15 and _cube2==24) or (_cube1==6 and _cube2==15):
                 self.LeftRotate(0,ANTICLOCKWISE)
            elif (_cube1==16 and _cube2==25) or (_cube1==7 and _cube2==16):
                 self.LeftRotate(1,ANTICLOCKWISE)
            elif (_cube1==17 and _cube2==26) or (_cube1==8 and _cube2==17):
                 self.LeftRotate(2,ANTICLOCKWISE)
            else:
                print("it is impossible")
        if _facetype==BOTTOM:
            if (_cube1==0 and _cube2==1) or (_cube1==1 and _cube2==2):
                 self.FrontRotate(0,ANTICLOCKWISE)
            elif (_cube1==9 and _cube2==10) or (_cube1==10 and _cube2==11):
                 self.FrontRotate(1,ANTICLOCKWISE)
            elif (_cube1==18 and _cube2==19) or (_cube1==19 and _cube2==20):
                 self.FrontRotate(2,ANTICLOCKWISE)
            elif (_cube1==1 and _cube2==0) or (_cube1==2 and _cube2==1):
                 self.FrontRotate(0,CLOCKWISE)
            elif (_cube1==10 and _cube2==9) or (_cube1==11 and _cube2==10):
                 self.FrontRotate(1,CLOCKWISE)
            elif (_cube1==19 and _cube2==18) or (_cube1==20 and _cube2==19):
                 self.FrontRotate(2,CLOCKWISE)
            #
            elif (_cube1==0 and _cube2==9) or (_cube1==9 and _cube2==18):
                 self.LeftRotate(0,CLOCKWISE)
            elif (_cube1==1 and _cube2==10) or (_cube1==10 and _cube2==19):
                 self.LeftRotate(1,CLOCKWISE)
            elif (_cube1==2 and _cube2==11) or (_cube1==11 and _cube2==20):
                 self.LeftRotate(2,CLOCKWISE)
            elif (_cube1==9 and _cube2==0) or (_cube1==18 and _cube2==9):
                 self.LeftRotate(0,ANTICLOCKWISE)
            elif (_cube1==10 and _cube2==1) or (_cube1==19 and _cube2==10):
                 self.LeftRotate(1,ANTICLOCKWISE)
            elif (_cube1==11 and _cube2==2) or (_cube1==20 and _cube2==11):
                 self.LeftRotate(2,ANTICLOCKWISE)
            else:
                print("it is impossible")
        if _facetype==RIGHT:
            if (_cube1==8 and _cube2==5) or (_cube1==5 and _cube2==2):
                 self.FrontRotate(0,CLOCKWISE)
            elif (_cube1==17 and _cube2==14) or (_cube1==14 and _cube2==11):
                 self.FrontRotate(1,CLOCKWISE)
            elif (_cube1==26 and _cube2==23) or (_cube1==23 and _cube2==20):
                 self.FrontRotate(2,CLOCKWISE)
            elif (_cube1==5 and _cube2==8) or (_cube1==2 and _cube2==5):
                 self.FrontRotate(0,ANTICLOCKWISE)
            elif (_cube1==14 and _cube2==17) or (_cube1==11 and _cube2==14):
                 self.FrontRotate(1,ANTICLOCKWISE)
            elif (_cube1==23 and _cube2==26) or (_cube1==20 and _cube2==23):
                 self.FrontRotate(2,ANTICLOCKWISE)
            #
            elif (_cube1==26 and _cube2==17) or (_cube1==17 and _cube2==8):
                 self.UpRotate(0,CLOCKWISE)
            elif (_cube1==23 and _cube2==14) or (_cube1==14 and _cube2==5):
                 self.UpRotate(1,CLOCKWISE)
            elif (_cube1==20 and _cube2==11) or (_cube1==11 and _cube2==2):
                 self.UpRotate(2,CLOCKWISE)
            elif (_cube1==17 and _cube2==26) or (_cube1==8 and _cube2==17):
                 self.UpRotate(0,ANTICLOCKWISE)
            elif (_cube1==14 and _cube2==23) or (_cube1==5 and _cube2==14):
                 self.UpRotate(1,ANTICLOCKWISE)
            elif (_cube1==11 and _cube2==20) or (_cube1==2 and _cube2==11):
                 self.UpRotate(2,ANTICLOCKWISE)
            else:
                print("it is impossible")
        if _facetype==BACK:
            if (_cube1==24 and _cube2==21) or (_cube1==21 and _cube2==18):
                 self.LeftRotate(0,ANTICLOCKWISE)
            elif (_cube1==25 and _cube2==22) or (_cube1==22 and _cube2==19):
                 self.LeftRotate(1,ANTICLOCKWISE)
            elif (_cube1==26 and _cube2==23) or (_cube1==23 and _cube2==20):
                 self.LeftRotate(2,ANTICLOCKWISE)
            elif (_cube1==21 and _cube2==24) or (_cube1==18 and _cube2==21):
                 self.FrontRotate(0,CLOCKWISE)
            elif (_cube1==22 and _cube2==25) or (_cube1==19 and _cube2==22):
                 self.FrontRotate(1,CLOCKWISE)
            elif (_cube1==23 and _cube2==26) or (_cube1==20 and _cube2==23):
                 self.FrontRotate(2,CLOCKWISE)
            #
            elif (_cube1==26 and _cube2==25) or (_cube1==25 and _cube2==24):
                 self.UpRotate(0,ANTICLOCKWISE)
            elif (_cube1==23 and _cube2==22) or (_cube1==22 and _cube2==21):
                 self.UpRotate(1,ANTICLOCKWISE)
            elif (_cube1==20 and _cube2==19) or (_cube1==19 and _cube2==18):
                 self.UpRotate(2,ANTICLOCKWISE)
            elif (_cube1==25 and _cube2==26) or (_cube1==24 and _cube2==25):
                 self.UpRotate(0,CLOCKWISE)
            elif (_cube1==22 and _cube2==23) or (_cube1==21 and _cube2==22):
                 self.UpRotate(1,CLOCKWISE)
            elif (_cube1==19 and _cube2==20) or (_cube1==18 and _cube2==19):
                 self.UpRotate(2,CLOCKWISE)
            else:
                print("it is impossible")

        


        

            
            
            

       





    def LeftDrag(self,x,y):
        self.trackball.mouseMotion(x,y)
        #print(self.trackball.getRotationMatrix())
        for i in self.cubesType2:
            i.applyRotation(self.trackball.getRotationMatrix())
        for i in self.cubes:
            i.applyRotation(self.trackball.getRotationMatrix())



    def release(self):
        self.trackball.mouseStopMotion()
    def test(self):
        self.UpRotate(0,ANTICLOCKWISE)
    def test2(self):
        self.FrontRotate(0,CLOCKWISE)
    def isRotating(self):
        for i in self.cubes:
            if i.isRotating()==True:
                return True
        return False

    def FrontRotate(self,platenNumber,direction_):
        if (not self.isRotating()):
            for i in self.cubeMove[platenNumber].flatten():
                self.cubes[i].rotateFront(direction_)
            self.Internal_FrontRotate(platenNumber,direction_)
        print(self.cubeMove)
    def LeftRotate(self,platenNumber,direction_):
        if (not self.isRotating()):
            for i in self.cubeMove[:,:,platenNumber].flatten():
                self.cubes[i].rotateLeft(direction_)
            self.Internal_LeftRotate(platenNumber,direction_)
        print(self.cubeMove)
            
    
    def UpRotate(self,platenNumber,direction_):
        if (not self.isRotating()):
            for i in self.cubeMove[:,2-platenNumber,:].flatten():
                self.cubes[i].rotateUp(direction_)
                print(i)
            self.Internal_Uprotate(platenNumber,direction_)
        print("doing uprotae now", self.cubeMove)
    def rotateRandomly(self,n):
        from random import choice
        from random import randint
        for _ in range(n):
            how=randint(0,2)
            plate=randint(0,2)
            dir_=choice([-1,1])
            if how==0:
                self.FrontRotate(plate,dir_)
            elif how==1:
                self.LeftRotate(plate,dir_)
            else:
                self.UpRotate(plate,dir_)
                

    def Internal_FrontRotate(self,n,direction_):
        temp=[]
        temp.append(self.cubeMove[n][0][0])
        temp.append(self.cubeMove[n][0][1])
        temp.append(self.cubeMove[n][0][2])
        temp.append(self.cubeMove[n][1][2])
        temp.append(self.cubeMove[n][2][2])
        temp.append(self.cubeMove[n][2][1])
        temp.append(self.cubeMove[n][2][0])
        temp.append(self.cubeMove[n][1][0])
        
        if (direction_== ANTICLOCKWISE):
           self.cubeMove[n][0][0]=temp[6]
           self.cubeMove[n][0][1]=temp[7]
           self.cubeMove[n][0][2]=temp[0]
           self.cubeMove[n][1][2]=temp[1]
           self.cubeMove[n][2][2]=temp[2]
           self.cubeMove[n][2][1]=temp[3]
           self.cubeMove[n][2][0]=temp[4]
           self.cubeMove[n][1][0]=temp[5]
        
        elif (direction_== CLOCKWISE):
        
           self.cubeMove[n][0][0]=temp[2]
           self.cubeMove[n][0][1]=temp[3]
           self.cubeMove[n][0][2]=temp[4]
           self.cubeMove[n][1][2]=temp[5]
           self.cubeMove[n][2][2]=temp[6]
           self.cubeMove[n][2][1]=temp[7]
           self.cubeMove[n][2][0]=temp[0]
           self.cubeMove[n][1][0]=temp[1]
            
            
        
        else:
            assert("something is wrong")


    def Internal_Uprotate(self,n,dir):

        temp=[]
        n=2-n
        temp.append(self.cubeMove[0][n][0])
        temp.append(self.cubeMove[0][n][1])
        temp.append(self.cubeMove[0][n][2])
        temp.append(self.cubeMove[1][n][2])
        temp.append(self.cubeMove[2][n][2])
        temp.append(self.cubeMove[2][n][1])
        temp.append(self.cubeMove[2][n][0])
        temp.append(self.cubeMove[1][n][0])
        
        if (dir== ANTICLOCKWISE):
           self.cubeMove[0][n][0]=temp[6]
           self.cubeMove[0][n][1]=temp[7]
           self.cubeMove[0][n][2]=temp[0]
           self.cubeMove[1][n][2]=temp[1]
           self.cubeMove[2][n][2]=temp[2]
           self.cubeMove[2][n][1]=temp[3]
           self.cubeMove[2][n][0]=temp[4]
           self.cubeMove[1][n][0]=temp[5]
        
        elif (dir==CLOCKWISE):
        
           self.cubeMove[0][n][0]=temp[2]
           self.cubeMove[0][n][1]=temp[3]
           self.cubeMove[0][n][2]=temp[4]
           self.cubeMove[1][n][2]=temp[5]
           self.cubeMove[2][n][2]=temp[6]
           self.cubeMove[2][n][1]=temp[7]
           self.cubeMove[2][n][0]=temp[0]
           self.cubeMove[1][n][0]=temp[1]
            
            
            
        
        else:
            assert("something is wrong")
        
    
    
    def Internal_LeftRotate(self,n,dir):

        temp=[]
        temp.append(self.cubeMove[2][2][n])
        temp.append(self.cubeMove[2][1][n])
        temp.append(self.cubeMove[2][0][n])
        temp.append(self.cubeMove[1][0][n])
        temp.append(self.cubeMove[0][0][n])
        temp.append(self.cubeMove[0][1][n])
        temp.append(self.cubeMove[0][2][n]) 
        temp.append(self.cubeMove[1][2][n])
        
        if (dir== CLOCKWISE):
           self.cubeMove[0][0][n]=temp[6]
           self.cubeMove[0][1][n]=temp[7]
           self.cubeMove[0][2][n]=temp[0]
           self.cubeMove[1][2][n]=temp[1]
           self.cubeMove[2][2][n]=temp[2]
           self.cubeMove[2][1][n]=temp[3]
           self.cubeMove[2][0][n]=temp[4]
           self.cubeMove[1][0][n]=temp[5]
        elif (dir== ANTICLOCKWISE):
           self.cubeMove[0][0][n]=temp[2]
           self.cubeMove[0][1][n]=temp[3]
           self.cubeMove[0][2][n]=temp[4]
           self.cubeMove[1][2][n]=temp[5]
           self.cubeMove[2][2][n]=temp[6]
           self.cubeMove[2][1][n]=temp[7]
           self.cubeMove[2][0][n]=temp[0]
           self.cubeMove[1][0][n]=temp[1]
        
        else:
            assert("something is wrong")
        
    












if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    #widget = MyWidget()
    #widget.show()
    widget=mainScreen()
    sys.exit(app.exec_())
    
