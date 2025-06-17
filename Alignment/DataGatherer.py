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
from devices_lib import Camera, Motor
import pyrbac

########Parameters and names#################

bgImages = []

japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
client = pyrbac.AuthenticationClient()
token = client.login_kerberos()
japc.rbacLogin(token=token)

##We should have most components ready, let's get data for training:
##BackLash
FoldingX = Motor('LHC.BSRH.FOLDING-H.B2', japc, 45) # todo
FoldingY = Motor('LHC.BSRH.FOLDING-V.B2', japc,0)

cam = Camera('LHC.BSRH.BTVDC-CAM.MAIN', japc)

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
