##Currently acquires data used to train a simple NN to align beam on the camera - both motors that are used here and the camera that is used here must be used for the alignment script, as the training values are unique to the motors/camera

########LIBRARIES############################
import pyjapc
from time import sleep
import csv
import random
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from devices_lib import Camera, Motor, Pneumatics
import pyrbac

########Parameters and names#################

bgImages = []

japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
client = pyrbac.AuthenticationClient()
token = client.login_kerberos()
japc.rbacLogin(token=token)

##We should have most components ready, let's get data for training:

##Please make sure that initially the targets are in OUT position (as we do not track whether they are in or out...)
## Note, currently there are no filters used, you need to move the filter wheel in order not to saturate the CM and you must remove the filter for the MonLS...

##Targets
TargetEA = Pneumatics('LHC.BSRH.IN-OUT.B2', japc, 2) # target1
TargetM3 = Motor('LHC.BSRH.TARGET-M3-TS100.B2', japc, 0, 0, 42661) # target 2?
# TargetCMx = Motor('LHC.BSRH.CMASK-X-TSZ100.B2', japc, 0, 60113, 40633)
# TargetCMy = Motor('LHC.BSRH.CMASK-Y-TSZ100.B2', japc, 0, 11940, 12040)
# TargetLS = Pneumatics('LHC.BSRH.IN-OUT.B2', japc, 3)

##Motors
M1x = Motor('LHC.BSRH.M1-X-TS50.B2', japc, 0) ##Carefull, moves only from 0 to 30mm (12000 steps)
M1Phi = Motor('LHC.BSRH.STEERING-H.B2', japc, 110) #Backlash not measured!!
M1Theta = Motor('LHC.BSRH.STEERING-V.B2', japc, 100) #Backlash not measured!!

# M3x = Motor('LHC.BSRH.M3-X-TSZ100.B2', japc, 0) #Backlash not measured!!
# M3y = Motor('LHC.BSRH.M3-Y-TSZ100.B2', japc, 0) #Backlash not measured!!
# M3Theta = Motor('LHC.BSRH.M3-THETA.B2', japc, 0)#Backlash not measured!!
# M3Phi = Motor('LHC.BSRH.M3-PHI.B2', japc, 0) #Backlash not measured!!
#
# FoldingX = Motor('LHC.BSRH.FOLDING-H.B2', japc, 45)
# FoldingY = Motor('LHC.BSRH.FOLDING-V.B2', japc,0)

##Cameras
MonEA = Camera('LHC.BSRH.MON.EA.B2', japc, TargetEA) # camera 1
# MonM3 = Camera('LHC.BSRH.MON.TGTM3.B2', japc, TargetM3) #Add target in case you want to look further than TGTM3 in Position M2
MonM3 = Camera('LHC.BSRH.MON.TGTM3.B2', japc) # camera 2
# MonCM = Camera('LHC.BSRH.CAM.CM.B2', japc, TargetCMx, TargetCMy)
# MonLS = Camera('LHC.BSRH.MON.LS.B2', japc, TargetLS)

#cam = Camera('LHC.BSRH.BTVDC-CAM.MAIN', japc)


################### for main cam #####################
MotorRangeX = 50 #How many steps the motor can make from reference to keep the image on the camera
MotorRangeY = 50


japc.startSubscriptions()
sleep(2)
##Save references so we can go back in case we lose the beam

refX = FoldingX.getPosition()
refY = FoldingY.getPosition()

PosX, PosY = cam.getBeamPosition()


##Open a csv file that we will use to save the training data:
TrainingFile = open('MainCamData.csv', 'a')
writeTraining = csv.writer(TrainingFile)
header = ["camX", "camY", "mX", "mY", "tX", "tY", "stepsX", "stepsY"]
writeTraining.writerow(header)

camX, camY = cam.getBeamPosition()
##Now let's generate training data:
for i in range(5000):
  if camX == -1 or camY == -1:
    print('Did not find beam, sending to references') ## This does not account for summing up the backlash...

    FoldingX.setRelPosition(refX-FoldingX.getPosition())
    FoldingY.setRelPosition(refY-FoldingY.getPosition())
    sleep(2)
    camX, camY = cam.getBeamPosition()
    if camX == -1 or camY == -1:
      print('Cannot find beam at reference location, stopping now')
      break
  
  mX = FoldingX.getPosition() #This might take long as it waits for motor to stop...
  mY = FoldingY.getPosition() #This might take long as it waits for motor to stop...
  
  stepsX = random.randint(-int(2*camX/968.0*MotorRangeX), int(2*(968.0-camX)*MotorRangeX/968.0), )
  stepsY = random.randint(-int(2*(608.0-camY)/608.0*MotorRangeY),int(2*camY/608.0*MotorRangeY))
  
  FoldingX.setRelPosition(stepsX)
  FoldingY.setRelPosition(stepsY)
  
  while FoldingX.isBusy() or FoldingY.isBusy():
    sleep(0.2)
  sleep(1)
  tX, tY = cam.getBeamPosition()
  
  if tX != -1 and tY != -1:
    #Prepare the row that will be written
    row = []
    row.append(camX)
    row.append(camY)
    row.append(mX)
    row.append(mY)
    row.append(tX)
    row.append(tY)
    row.append(stepsX)
    row.append(stepsY)
    
    writeTraining.writerow(row)
    
  camX = tX #what was target is 
  camY = tY
  print('Done {} movements. '.format(i), end = '\r')


FoldingX.setAbsPosition(refX)
FoldingY.setAbsPosition(refY)
japc.stopSubscriptions()

TrainingFile.close()

################### for main cam end #####################

def get_steps():
  
  return 0, 0, 0

def move_motors(stepsX, stepsPhi, stepsTheta):

  M1x.setRelPosition(stepsX)
  M1Phi.setRelPosition(stepsPhi)
  M1Theta.setRelPosition(stepsTheta)

  while M1x.isBusy() or M1Phi.isBusy() or M1Theta.isBusy():
    sleep(2)
  sleep(1)

def measure_c1():
  cam1X, cam1Y = MonEA.getBeamPosition()
  if cam1X == -1 or cam1Y == -1:
    print('Did not find beam, sending to references')  ## This does not account for summing up the backlash...
    M1x.setAbsPosition(ref_x)
    M1Phi.setAbsPosition(ref_phi)
    M1Theta.setAbsPosition(ref_theta)
    sleep(2)
    cam1X, cam1Y = MonEA.getBeamPosition()
    if cam1X == -1 or cam1Y == -1:
      print('Cannot find beam at reference location, stopping now')
  return cam1X, cam1Y

def measure_c2():
  cam2X, cam2Y = MonM3.getBeamPosition()
  if cam2X == -1 or cam2Y == -1: # not sure I can do this for cam2
    print('Did not find beam, sending to references')  ## This does not account for summing up the backlash...
    M1x.setAbsPosition(ref_x)
    M1Phi.setAbsPosition(ref_phi)
    M1Theta.setAbsPosition(ref_theta)
    sleep(2)
    cam2X, cam2Y = MonM3.getBeamPosition()
    if cam2X == -1 or cam2Y == -1:
      print('Cannot find beam at reference location, stopping now')
  return cam2X, cam2Y


# How many steps the motor can make from reference to keep the image on the camera
motor_range_x = 15 # is it in mm?
motor_range_phi = 43000 # in steps
motor_range_theta = 27241 # in steps

japc.startSubscriptions()
sleep(2)

ref_x = M1x.getPosition()
ref_phi = M1Phi.getPosition()
ref_theta = M1Theta.getPosition()

# target 1 IN at the start
PosX, PosY = MonEA.getBeamPosition()

##Open a csv file that we will use to save the training data:
TrainingFile = open('target1_target2_data.csv', 'a')
writeTraining = csv.writer(TrainingFile)
header = ["cam1X", "cam1Y", "cam2X", "cam2Y", "mX", "mPhi", "mTheta", "t1X", "t1Y", "t2X", "t2Y", "stepsX", "stepsPhi", "stepsTheta"]
writeTraining.writerow(header)


iterations = 20

# make sure target 1 is IN
if TargetEA.getPosition() == 0: # 0 or 1 is IN/OUT
  TargetEA.changePosition()

for i in range(iterations):

  cam1X, cam1Y = measure_c1()

  mX = M1x.getPosition()
  mPhi = M1Phi.getPosition()
  mTheta = M1Theta.getPosition()

  TargetEA.changePosition()

  cam2X, cam2Y = measure_c2()

  stepsX, stepsPhi, stepsTheta = get_steps()
  move_motors(stepsX, stepsPhi, stepsTheta)

  t2X, t2Y = MonM3.getBeamPosition()

  TargetEA.changePosition()

  t1X, t1Y = MonEA.getBeamPosition()

  if t1X != -1 or t1Y != -1 or t2X != -1 or t2Y != -1:
    row = []
    row.append(cam1X)
    row.append(cam1Y)
    row.append(cam2X)
    row.append(cam2Y)
    row.append(mX)
    row.append(mPhi)
    row.append(mTheta)
    row.append(t1X)
    row.append(t1Y)
    row.append(t2X)
    row.append(t2Y)
    row.append(stepsX)
    row.append(stepsPhi)
    row.append(stepsTheta)

    writeTraining.writerow(row)

  # cam1X = t1X
  # cam1Y = t1Y
  print('Done {} movements. '.format(i), end='\r')


M1x.setAbsPosition(ref_x)
M1Phi.setAbsPosition(ref_phi)
M1Theta.setAbsPosition(ref_theta)

japc.stopSubscriptions()
TrainingFile.close()