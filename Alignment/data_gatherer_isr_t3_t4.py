# gather data on isr - mirror2 (x, y, phi, theta) + target3 (see-through) + target4

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

bgImages = []

japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
client = pyrbac.AuthenticationClient()
token = client.login_kerberos()
japc.rbacLogin(token=token)

# ISR Motors
motor_x = Motor('TST.cfv-375-bsrdev.mot.3-2', japc, 5)
motor_y = Motor('TST.cfv-375-bsrdev.mot.3-3', japc, 5)
motor_phi = Motor('TST.cfv-375-bsrdev.mot.3-4', japc, 168)
motor_theta = Motor('TST.cfv-375-bsrdev.mot.3-5', japc, 125)

# ISR Cameras
cam_1 = Camera("TST.BSRH.MON.M2", japc)
cam_2 = Camera("TST.BSRH.CAM.CM", japc)

# How many steps the motor can make from reference to keep the image on the camera
motor_range_x_t1 = 0 # to be measured
motor_range_y_t1 = 0
motor_range_phi_t1 = 0
motor_range_theta_t1 = 0

motor_range_x_t2 = 0
motor_range_y_t2 = 0
motor_range_phi_t2 = 0
motor_range_theta_t2 = 0

japc.startSubscriptions()
sleep(2)

# reference position to return to if something goes wrong
ref_x = motor_x.getPosition()
ref_y = motor_y.getPosition()
ref_phi = motor_phi.getPosition()
ref_theta = motor_theta.getPosition()

# open a csv file that we will use to save the training data:
TrainingFile = open('isr_t3_t4_data.csv', 'a')
writeTraining = csv.writer(TrainingFile)
header = ["cam1X", "cam1Y", "cam2X", "cam2Y", "mX", "mY", "mPhi", "mTheta", "t1X", "t1Y", "t2X", "t2Y", "stepsX", "stepsY", "stepsPhi", "stepsTheta"]
writeTraining.writerow(header)

iterations = 20

# collect data


motor_x.setAbsPosition(ref_x)
motor_y.setAbsPosition(ref_y)
motor_phi.setAbsPosition(ref_phi)
motor_theta.setAbsPosition(ref_theta)

japc.stopSubscriptions()
TrainingFile.close()