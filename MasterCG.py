## So this is it!
from config import CoronagraphDevices
from devices_lib import Camera, Motor, Pneumatics, FWheel
from aligners_lib import T1T2AlignmentNN, T3T4AlignmentNN, T5T6AlignmentNN
import pyjapc
import pyrbac
import numpy as np
from time import sleep
import time
import os

import matplotlib.pyplot as plt

# import acquisition_lib
# import ??

# Start with putting all in, then check the position on each camera and align only the piece that is necessary...

#We need 2 japc instances to separate the subscriptions
client = pyrbac.AuthenticationClient()
token = client.login_kerberos()

japcTg = pyjapc.PyJapc(selector = '', incaAcceleratorName = '')
japcTg.rbacLogin(token=token)


# 'LHC.BCCM.B2.A/SummaryAcquisition#beamEnergy' # Variable of the LHC's energy - 5.5TeV to start...

#japcAcq = pyjapc.PyJapc(selector = '', incaAcceleratorName = '')
#japcAcq.rbacLogin(token = token)


# Here we create instances... Targets, Cameras within aligners,
Targets = CoronagraphDevices(japcTg)

Cam1 = Camera('LHC.BSRH.MON.EA.B2', japcTg)
Cam2 = Camera('LHC.BSRH.MON.TGTM3.B2', japcTg)
Cam4 = Camera('LHC.BSRH.CAM.CM.B2', japcTg)
Cam5 = Camera('LHC.BSRH.MON.LS.B2', japcTg)
# CamMain = Camera('LHC.BSRH.BTVDC-CAM.MAIN.B2', japcAcq)


# MaskX = Motor("LHC.BSRH.CMASK-X-TSZ-100.B2", japcAcq) # This might be stupid, but that's ok
# MaskY = Motor("LHC.BSRH.CMASK-Y-TSZ-100.B2", japcAcq)

japcTg.startSubscriptions()
sleep(2)
Targets.allIn()

#Here we get bg for all static cameras... using the proper exposure time...
## Exposure times according to the total charge in the B2:
chargeB2 = japcTg.getParam('LHC.BCTFR.A6R4.B2/Acquisition#wholeBeamIntensity') # Normalizing by the full machine at 3.5e14
try:
    timingFactor = np.round(3.5e14/chargeB2, 2)
except:
    timingFactor = 1

Cam1.changeExpTime(timingFactor*150)
Cam2.changeExpTime(timingFactor*50)
Cam4.changeExpTime(timingFactor*500) # With ND4 in...
Cam5.changeExpTime(timingFactor*300) # No ND - so rescale by ND....

sleep(15) # Wait for all exposure times to be set...

bgT2 = np.asarray(Cam2.captureImage(True))
bgT4 = np.asarray(Cam4.captureImage(True))
bgT5 = np.asarray(Cam5.captureImage(True))

# Let's save References into Data/References/year/month/day
DATA_PATH_ROOT = '/user/jpucek/BSRH/Data/References'
ts = time.strftime('%Y_%m_%d', time.localtime())
path = os.path.join(DATA_PATH_ROOT, time.strftime('%Y/%m/%d/', time.localtime()))
os.makedirs(path, exist_ok=True)

cam1_tstamp = 'cam1_' + time.strftime('%H_%M_%S.png', time.localtime())
cam2_tstamp = 'cam2_' + time.strftime('%H_%M_%S.png', time.localtime())
cam3_tstamp = 'cam3_' + time.strftime('%H_%M_%S.png', time.localtime())
cam4_tstamp = 'cam4_' + time.strftime('%H_%M_%S.png', time.localtime())
cam5_tstamp = 'cam5_' + time.strftime('%H_%M_%S.png', time.localtime())

cam1_file = os.path.join(path, cam1_tstamp)
cam2_file = os.path.join(path, cam2_tstamp)
cam3_file = os.path.join(path, cam3_tstamp)
cam4_file = os.path.join(path, cam4_tstamp)
cam5_file = os.path.join(path, cam5_tstamp)

print('Measuring cam 1')
cam1_image = Cam1.captureImage()
Targets.measureT2()
print('Measuring cam 2')
cam2_image = Cam2.captureImage() - bgT2
Targets.measureT3()
print('Measuring cam 3')
cam3_image = Cam2.captureImage() - bgT2
Targets.measureT4()
print('Measuring cam 4')
cam4_image = Cam4.captureImage() - bgT4
Targets.measureT5()
print('Measuring cam 5')
cam5_image = Cam5.captureImage() - bgT5

print('Saving images')
plt.imsave(cam1_file, cam1_image, cmap = 'jet', vmin = 0, vmax = np.median(cam1_image)) #There is no bg subtraction for this one...
plt.imsave(cam2_file, cam2_image, cmap = 'jet', vmin = 0, vmax = np.median(cam2_image))
plt.imsave(cam3_file, cam3_image, cmap = 'jet', vmin = 0, vmax = np.median(cam3_image))
plt.imsave(cam4_file, cam4_image, cmap = 'jet', vmin = 0, vmax = np.median(cam4_image))
plt.imsave(cam5_file, cam5_image, cmap = 'jet', vmin = 0, vmax = np.median(cam5_image))
print('Images saved')
# Now we have saved references we can align:
# TODO alignment

Targets.measureT6() #This puts in the ND4...
japcTg.stopSubscriptions() # No more target movement until next fill (hopefully)

sleep(2)