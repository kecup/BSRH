## Script to perform automatic alignment of the BSRH - up to the LS as the final focus is pretty difficult to do...
#First get motor positions of everything that we will move
#Then capture the beam on all targets

########LIBRARIES############################
import pyjapc
import pyrbac
import datetime as dt
import numpy as np
from scipy.optimize import curve_fit
from time import sleep
import csv
import random

import sys
import select

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

import joblib

########Parameters and names#################

bgImages = []

japc = pyjapc.PyJapc(selector = '', incaAcceleratorName = '')
client = pyrbac.AuthenticationClient()
token = client.login_kerberos()
japc.rbacLogin(token=token)

def gaussian(x, amp, cen, wid, const):
  return amp*np.exp(-(x-cen)**2/(2*wid**2))+const

#def CheckSystem():#Function which will check all coms and acq statuses
#  for cam in Cameras:
#    if japc.getParam(cam+'/Acquisition#AcqOK'):
#      japc.getParam(cam+'/Acquisition#image1_clkcnt')

class Motor:
  def __init__(self, name, backlash_steps = 0):
    self.name = name
    self.position = -1
    self.lastMotionRel = 0
    self.backlash_steps = backlash_steps
    try:
      if japc.getParam(self.name+'/Acquisition#motor_busy') == False:
        self.position = japc.getParam(self.name+'/Acquisition#position') #This is just to test that acquisition can be made
      else:
        print("Motor is moving!")
    except:
      print('Invalid name!')
  def isBusy(self):
    return japc.getParam(self.name+'/Acquisition#motor_busy')
 
  #def getPosition(self): ## This is OLD
  #  if japc.getParam(self.name+'/Acquisition#motor_busy') == False:
  #    self.position = japc.getParam(self.name+'/Acquisition#position_counter') #Uses position in steps! So carefull with movements...
  #  else:
  #    self.position = -1
  #    print('Motor still busy!!')
  #  return self.position
  
  def getPosition(self): ## This is NEW
    while japc.getParam(self.name+'/Acquisition#motor_busy') == True:
      sleep(0.5)
    self.position = japc.getParam(self.name+'/Acquisition#position_counter') #Uses position in steps! So carefull with movements...
    return self.position
   
  def setAbsPosition(self, absPosition): #This is motion in absolute raw steps! No backlash taken into account!!!
    if japc.getParam(self.name+'/Acquisition#motor_busy') == False:
      mv = {'value': absPosition, 'mode': 2, 'units': 2}
      japc.setParam(self.name+'/Move', mv)
    else:
      print('Cannot comply, motor is busy!')
   
  def setRelPosition(self, relPosition): #Motion in relative steps!
    if japc.getParam(self.name+'/Acquisition#motor_busy') == False:  
      
      if relPosition*self.lastMotionRel < 0: #Change of sign
        self.lastMotionRel = relPosition
        relPosition = relPosition + np.sign(relPosition)*self.backlash_steps
      else:
        self.lastMotionRel = relPosition
      
      mv = {'value': relPosition, 'mode': 1, 'units': 2}
      japc.setParam(self.name+'/Move', mv)
      
    else:
      print('Cannot comply, motor is busy!')

class Camera:
  def __init__(self, name, inPos = None, outPos = None, motorX = None, motorY = None): ## Provide motors as Class and positions in steps...
    self.name = name
    self.motorX = motorX
    self.motorY = motorY
    self.inPos = inPos
    self.outPos = outPos
    self.updated = dt.datetime.now()
    self.image = []
    if inPos != None and outPos != None:
      self.movableAxis = len(inPos)
    else:
      self.movableAxis = 0
    self.isIn = -1
    self.subscription = japc.subscribeParam(self.name+'/LastImage#image2D', self.updateImage)
    
  def updateImage(self, parameter, value):
    self.image = value  
    self.updated = dt.datetime.now()
    
  def checkIn(self):
    output = False
    if (self.movableAxis > 0 and motorX.getPosition() == inPos[0]) and (self.movableAxis == 1 or motorY.getPosition() == inPos[1]) or self.movableAxis == 0:
      output = True
    return output
  
  def captureImage(self, MoveOut = False):## Should think whether subscription is the way forward or not...
    lastUpdate = self.updated
    if self.checkIn() == False: #Checking if the camera/target is in.
      ##This means that we have to move motor(s)
      motorX.setAbsPosition(inPos[0]) #at least one motor must be present
      if movableAxis == 2:
        motorY.setAbsPosition(inPos[1])
        while motorX.isBusy() or motorY.isBusy():
          sleep(0.2)
      while motorX.isBusy():
        sleep(0.2)
    
    cycle = 0
    while lastUpdate == self.updated and cycle < 10:
      sleep(0.2)
      cycle = cycle + 1
      
    if cycle >= 10:
      print('Getting image timed-out.')
      outImage = []
    else:
      outImage = self.image
    
    ##Testing to send the position out again?
    if self.movableAxis>0 and MoveOut == True:
      motorX.setAbsPosition(outPos[0]) #at least one motor must be present
      if movableAxis == 2:
        motorY.setAbsPosition(outPos[1])
       
    return outImage

  def getBeamPosition(self):
    ##Get an image that we will analyse:
    image = self.captureImage()
    ##Perform a 1D fit of projections:
    try:
      projection_x = np.sum(image, axis=0)
      projection_y = np.sum(image, axis=1)
    except:
      projection_x = []
      projection_y = []
      
    ##Prepare bounds for fits:
    gBounds = ([1e4, 0, 1, 10],[1e7, 960, 100, 2e4])
    
    try:
      #resX = [np.max(projection_x), np.argmax(projection_x)]
      resX, _ = curve_fit(gaussian, range(len(projection_x)), projection_x, (6e4, np.argmax(projection_x), 10.0, 300.0), bounds = gBounds)
      if resX[0] < 2e4:
        resX[1] = -1      
    except:
      print("Did not fit in X.")
      resX = [-1, -1, -1]


    try:
      #resY = [np.max(projection_y), np.argmax(projection_y)]
      resY, _ = curve_fit(gaussian, range(len(projection_y)), projection_y, (6e4, np.argmax(projection_y), 10.0, 300.0), bounds = gBounds)
      if resY[0] < 2e4:
        resY[1] = -1
    except:
      print("Did not fit in Y.")
      resY = [-1, -1, -1]
    
    return resX[1], resY[1]
    
class FeedbackNN(nn.Module): #Not used here
  def __init__(self, state_size, action_size, seed = 45) -> None:
      self.NNsize = 16
      super(FeedbackNN, self).__init__()
      self.seed = torch.manual_seed(seed)
      self.fc1 = nn.Linear(state_size, self.NNsize) #First Neuron layer
      self.fc2 = nn.Linear(self.NNsize,self.NNsize)#Next Neuron layer
      self.fc3 = nn.Linear(self.NNsize,self.NNsize)#Next Neuron layer
      self.fc4 = nn.Linear(self.NNsize, action_size)#Last Neuron layer

  def forward(self, state):
      x = self.fc1(state) #output of the first layer
      x = F.relu(x) # rectify function of the output

      y = self.fc2(x)
      y = F.relu(y)
      
      #y = self.fc3(y)
      #y = F.relu(y)
      
      z = self.fc4(y)
      return z


class BeamSteerer:
    def __init__(self, forward_model: nn.Module, lr=0.2, iters=30):
        """
        forward_model: a trained nn.Module mapping (x,u)→x_next
        lr: inner‐loop step size on u
        iters: number of GD steps per control cycle
        """
        self.f_phi = forward_model
        self.lr = lr
        self.iters = iters

    def compute_control(self, x0: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        x0: tensor([x,y]), target: tensor([x_t,y_t])
        returns: optimized u (tensor([Δx,Δy]))
        """
        # initialize u as a small zero vector (or reuse from last cycle)
        u = torch.zeros(2, requires_grad=True)

        opt = optim.SGD([u], lr=self.lr)
        for _ in range(self.iters):
            opt.zero_grad()
            xu = torch.cat([x0, u], dim=0).unsqueeze(0)
            x_pred = self.f_phi(xu).squeeze(0)          # predict next beam pos
            loss = nn.functional.mse_loss(x_pred, target)
            loss.backward()
            opt.step()

        # detach u to get a plain tensor with no gradient
        return u.detach()

    def step(self, motorX, motorY, camera, target):
        # 1) read current beam
        x0_np = camera.getBeamPosition()       # returns e.g. [x,y] numpy
        x0 = torch.tensor(x0_np, dtype=torch.float32)

        # 2) compute optimal motor move
        u_opt = self.compute_control(x0, torch.tensor(target, dtype = torch.float32))

        # 3) send to hardware
        motion = u_opt.tolist()
        
        if x0_np[0] == -1 or x0_np[1] == -1: #Do not move if you don't see the beam
          motion = [0,0]
        motorX.setRelPosition(motion[0])
        motorY.setRelPosition(motion[1])
      
##We should have most components ready, let's get data for training:
##BackLash

M3X = Motor('LHC.BSRH.FOLDING-H.B2', 45)  #LHC
M3Y = Motor('LHC.BSRH.FOLDING-V.B2', 0) #LHC

cam1 = Camera('LHC.BSRH.BTVDC-CAM.MAIN') #LHC

target = [484, 304]

##Now load our NN:

model = FeedbackNN(4,2)
model.load_state_dict(torch.load('MainCamModel.pth'))
model.eval()

steerer = BeamSteerer(model, lr = 0.1, iters = 20)

japc.startSubscriptions()

M3X.setRelPosition(10)
M3Y.setRelPosition(10)

count = 0
while True:

  if select.select([sys.stdin], [], [], 0.0)[0]:
    print("Key pressed! Exiting loop.")
    break
  
  camX, camY = cam1.getBeamPosition()
  if camX == -1 and camY == -1: # Any other way we can implement not doing anything unless we have a nice beam?
    print('Could not find the beam, stopping alignment')
    break
  
  if count == 10:
    print("Distance to target = ", np.round(np.sqrt((camX-target[0])**2+(camY-target[1])**2),2))
    count = 0
  if np.abs(camX-target[0]) > 5.0 and np.abs(camY-target[1]) > 5.0:
    steerer.step(M3X, M3Y, cam1, target)
    print('Moving, distance to target = ', np.round(np.sqrt((camX-target[0])**2+(camY-target[1])**2),2), dt.datetime.now())
    sleep(2)
  
  sleep(1)
  count = count + 1
japc.stopSubscriptions()
print('')
#At some point we have to call japc.startSubscriptions()!!
