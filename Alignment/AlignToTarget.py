## Script to perform automatic alignment of the BSRH - up to the LS as the final focus is pretty difficult to do...
#First get motor positions of everything that we will move
#Then capture the beam on all targets

########LIBRARIES############################
import pyjapc
import pyrbac
import numpy as np
from time import sleep
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import select
from devices_lib import Camera, Motor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

########Parameters and names#################

bgImages = []

japc = pyjapc.PyJapc(selector = '', incaAcceleratorName = '')
client = pyrbac.AuthenticationClient()
token = client.login_kerberos()
japc.rbacLogin(token=token)


    
class FeedbackNN(nn.Module): #Not used here
  def __init__(self, state_size, action_size, seed = 45) -> None:
      self.NNsize = 16
      super(FeedbackNN, self).__init__()
      self.seed = torch.manual_seed(seed)
      self.fc1 = nn.Linear(state_size, self.NNsize) # First Neuron layer
      self.fc2 = nn.Linear(self.NNsize,self.NNsize) # Next Neuron layer
      self.fc3 = nn.Linear(self.NNsize,self.NNsize) # Next Neuron layer
      self.fc4 = nn.Linear(self.NNsize, action_size) # Last Neuron layer

  def forward(self, state):
      x = self.fc1(state) # output of the first layer
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
        motorX.setRelPosition(motion[0])
        motorY.setRelPosition(motion[1])
      
##We should have most components ready, let's get data for training:
##BackLash
M3X = Motor('TST.cfv-375-bsrdev.mot.3-4', japc, 168) #ISR
M3Y = Motor('TST.cfv-375-bsrdev.mot.3-5', japc, 125) #ISR

# M3X = Motor('LHC.BSRH.M3-PHI.B2', 35)  #LHC
# M3Y = Motor('LHC.BSRH.M3-THETA.B2', 30) #LHC

cam1 = Camera('TST.BSRH.CAM.CM', japc) #ISR
# cam1 = Camera('LHC.BSRH.CAM.CM.B2') #LHC

target = [484, 304]

##Now load our NN:

model = FeedbackNN(4,2)
model.load_state_dict(torch.load('CM_feedbackModel.pth'))
model.eval()

steerer = BeamSteerer(model, lr = 0.1, iters = 20)

japc.startSubscriptions()

M3X.setRelPosition(30)
M3Y.setRelPosition(30)


done = False
while True:

  if select.select([sys.stdin], [], [], 0.0)[0]:
        print("Key pressed! Exiting loop.")
        break
  
  camX, camY = cam1.getBeamPosition()
  if np.abs(camX-target[0]) > 5 and np.abs(camY-target[1]) > 5:
    steerer.step(M3X, M3Y, cam1, target)
    print('Moving, distance to target = ', np.sqrt((camX-target[0])**2+(camY-target[1])**2))
    sleep(2)
  sleep(1)
  
japc.stopSubscriptions()
print('')
#At some point we have to call japc.startSubscriptions()!!
