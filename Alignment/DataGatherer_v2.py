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
TargetEA = Pneumatics('LHC.BSRH.IN-OUT.B2', japc, 2)
TargetM3 = Motor('LHC.BSRH.TARGET-M3-TS100.B2', japc, 0, 0, 42661)
TargetCMx = Motor('LHC.BSRH.CMASK-X-TSZ100.B2', japc, 0, 60113, 40633)
TargetCMy = Motor('LHC.BSRH.CMASK-Y-TSZ100.B2', japc, 0, 11940, 12040)
TargetLS = Pneumatics('LHC.BSRH.IN-OUT.B2', japc, 3)

##Motors
M1x = Motor('LHC.BSRH.M1-X-TS50.B2', japc, 0) ##Carefull, moves only from 0 to 30mm (12000 steps)
M1Phi = Motor('LHC.BSRH.STEERING-H.B2', japc, 110) #Backlash not measured!!
M1Theta = Motor('LHC.BSRH.STEERING-V.B2', japc, 100) #Backlash not measured!!

M3x = Motor('LHC.BSRH.M3-X-TSZ100.B2', japc, 0) #Backlash not measured!!
M3y = Motor('LHC.BSRH.M3-Y-TSZ100.B2', japc, 0) #Backlash not measured!!
M3Theta = Motor('LHC.BSRH.M3-THETA.B2', japc, 0)#Backlash not measured!!
M3Phi = Motor('LHC.BSRH.M3-PHI.B2', japc, 0) #Backlash not measured!!

FoldingX = Motor('LHC.BSRH.FOLDING-H.B2', japc, 45)
FoldingY = Motor('LHC.BSRH.FOLDING-V.B2', japc,0)

##Cameras
MonEA = Camera('LHC.BSRH.MON.EA.B2', japc, TargetEA)
MonM3 = Camera('LHC.BSRH.MON.TGTM3.B2', japc, TargetM3) #Add target in case you want to look further than TGTM3 in Position M2
MonCM = Camera('LHC.BSRH.CAM.CM.B2', japc, TargetCMx, TargetCMy)
MonLS = Camera('LHC.BSRH.MON.LS.B2', japc, TargetLS)

#cam = Camera('LHC.BSRH.BTVDC-CAM.MAIN', japc)



MotorRangeX = 50#How many steps the motor can make from reference to keep the image on the camera
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
print('')
# print(i)
#At some point we have to call japc.startSubscriptions()!!
