##Currently acquires data used to train a simple NN to align beam on the camera - both motors that are used here and the camera that is used here must be used for the alignment script, as the training values are unique to the motors/camera

########LIBRARIES############################
import pyjapc
from time import sleep
import csv
import random
import sys
import os
import math

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
TargetEA = Pneumatics('LHC.BSRH.IN-OUT.B2', japc, 1) # target1
TargetM3 = Motor('LHC.BSRH.TARGET-M3-TS100.B2', japc, 0, 0, 42661) # target 2?
# TargetCMx = Motor('LHC.BSRH.CMASK-X-TSZ100.B2', japc, 0, 60113, 40633)
# TargetCMy = Motor('LHC.BSRH.CMASK-Y-TSZ100.B2', japc, 0, 11940, 12040)
# TargetLS = Pneumatics('LHC.BSRH.IN-OUT.B2', japc, 2)

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
MonEA = Camera('LHC.BSRH.MON.EA.B2', japc, motorX=TargetEA, returnTarget=False) # camera 1
# MonM3 = Camera('LHC.BSRH.MON.TGTM3.B2', japc, TargetM3) #Add target in case you want to look further than TGTM3 in Position M2
MonM3 = Camera('LHC.BSRH.MON.TGTM3.B2', japc) # camera 2
# MonCM = Camera('LHC.BSRH.CAM.CM.B2', japc, TargetCMx, TargetCMy)
# MonLS = Camera('LHC.BSRH.MON.LS.B2', japc, TargetLS)

#cam = Camera('LHC.BSRH.BTVDC-CAM.MAIN', japc)


def get_steps(cam_width=968.0, cam_height=608.0):
  # this needs some brain cells

  steps_x = random.randint(
    -int(2 * cam2X / cam_width * motor_range_x_t2),
    int(2 * (cam_width - cam2X) / cam_width * motor_range_x_t2)
  )

  steps_phi = random.randint(
    -int(2 * cam2X / cam_width * motor_range_phi_t2),
    int(2 * (cam_width - cam2X) / cam_width * motor_range_phi_t2)
  )

  steps_theta = random.randint(
    -int(2 * (cam_height - cam2Y) / cam_height * motor_range_theta_t2),
    int(2 * cam2Y / cam_height * motor_range_theta_t2)
  )

  return steps_x, steps_phi, steps_theta

def get_steps_v2():
  steps_x = (random.randint(M1x.getPosition()-motor_range_x_t2, M1x.getPosition()+motor_range_x_t2))-M1x.getPosition()

  steps_phi = (random.randint(M1Phi.getPosition() - motor_range_phi_t2,
                            M1Phi.getPosition() + motor_range_phi_t2)) - M1Phi.getPosition()

  steps_theta = (random.randint(M1Theta.getPosition() - motor_range_theta_t2,
                            M1Theta.getPosition() + motor_range_theta_t2)) - M1Theta.getPosition()

  return steps_x, steps_phi, steps_theta

def move_motors(stepsX, stepsPhi, stepsTheta):
  """
  Moves all three motors by their respective steps and waits for them to move there.
  """
  M1x.setRelPosition(stepsX)
  M1Phi.setRelPosition(stepsPhi)
  M1Theta.setRelPosition(stepsTheta)

  # wait for motors to move
  counter = 0
  while M1x.isBusy() or M1Phi.isBusy() or M1Theta.isBusy():
    sleep(2)
    counter += 1
    if counter > 5:
      break
  sleep(1)

def measure_c1():
  """
  Measure camera 1 data.
  """
  cam1X, cam1Y = MonEA.getBeamPosition()
  if cam1X == -1 or cam1Y == -1:
    print('Did not find beam, sending to references')  ## This does not account for summing up the backlash...
    sleep(2)
    M1x.setAbsPosition(ref_x)
    M1Phi.setAbsPosition(ref_phi)
    M1Theta.setAbsPosition(ref_theta)
    sleep(8)
    cam1X, cam1Y = MonEA.getBeamPosition()
    if cam1X == -1 or cam1Y == -1:
      print('Cannot find beam at reference location, stopping now')
  return cam1X, cam1Y

def measure_c2():
  """
  Measure camera 2 data.
  """
  cam2X, cam2Y = MonM3.getBeamPosition()
  if cam2X == -1 or cam2Y == -1: # not sure I can do this for cam2
    print('Did not find beam, sending to references')  ## This does not account for summing up the backlash...
    sleep(2)
    M1x.setAbsPosition(ref_x)
    M1Phi.setAbsPosition(ref_phi)
    M1Theta.setAbsPosition(ref_theta)
    sleep(8)
    cam2X, cam2Y = MonM3.getBeamPosition()
    if cam2X == -1 or cam2Y == -1:
      print('Cannot find beam at reference location, stopping now')
  return cam2X, cam2Y


# How many steps the motor can make from reference to keep the image on the camera
motor_range_x_t1 = 6000 # in steps (15mm)
motor_range_phi_t1 = 43000 # in steps
motor_range_theta_t1 = 27241 # in steps

motor_range_x_t2 = 6000# in steps (15mm)
motor_range_phi_t2 = 5000 # in steps
motor_range_theta_t2 = 3000 # in steps

japc.startSubscriptions()
sleep(2)

# reference position to return to if something goes wrong
ref_x = M1x.getPosition()
ref_phi = M1Phi.getPosition()
ref_theta = M1Theta.getPosition()

# open a csv file that we will use to save the training data:
TrainingFile = open('lhc_t1_t2_data_v2.csv', 'a')
writeTraining = csv.writer(TrainingFile)
header = ["cam1X", "cam1Y", "cam2X", "cam2Y", "mX", "mPhi", "mTheta", "t1X", "t1Y", "t2X", "t2Y", "stepsX", "stepsPhi", "stepsTheta"]
writeTraining.writerow(header)

iterations = 2000

# make sure target 1 is IN
if TargetEA.getPosition() == 0: # 0 is OUT and 1 is IN
  TargetEA.changePosition()
  sleep(3)

for i in range(iterations):
  print("Starting iteration " + str(i+1))
  cam1X, cam1Y = measure_c1()

  mX = M1x.getPosition()
  mPhi = M1Phi.getPosition()
  mTheta = M1Theta.getPosition()

  # print("Identifikator 1: ", MonEA.motorX.isIn())
  MonEA.motorX.changePosition()
  sleep(2)
  cam2X, cam2Y = measure_c2()

  stepsX, stepsPhi, stepsTheta = get_steps_v2()
  move_motors(stepsX, stepsPhi, stepsTheta)

  t2X, t2Y = MonM3.getBeamPosition()

  t1X, t1Y = MonEA.getBeamPosition()

  if not math.isclose(t1X, -1.0, rel_tol=1e-9) or not math.isclose(t1Y, -1.0, rel_tol=1e-9) or not math.isclose(t2X, -1.0, rel_tol=1e-9) or not math.isclose(t2Y, -1.0, rel_tol=1e-9):
    row = [cam1X, cam1Y, cam2X, cam2Y, mX, mPhi, mTheta, t1X, t1Y, t2X, t2Y, stepsX, stepsPhi, stepsTheta]
    writeTraining.writerow(row)
    TrainingFile.flush()


  # cam1X = t1X
  # cam1Y = t1Y
  print('Done {} movements. '.format(i), end='\r')


M1x.setAbsPosition(ref_x)
M1Phi.setAbsPosition(ref_phi)
M1Theta.setAbsPosition(ref_theta)

japc.stopSubscriptions()
TrainingFile.close()
