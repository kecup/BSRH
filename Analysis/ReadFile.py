import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
import math
from scipy import fft
from scipy.stats import skewnorm
from scipy.stats import exponnorm
from scipy.optimize import curve_fit
import os
import math
import lmfit
import dill as pickle
import pyjapc
import zstandard as zstd
import datetime as dt

### This script opens the data acquired by the BSRH (+ additional devices), sample to be change accordingly
#### Run in interactive mode...

japc = pyjapc.PyJapc(selector='', incaAcceleratorName='') #Stupid Java VM must be running to open the java values... We have to retype this...

dctx = zstd.ZstdDecompressor()
directory = '/user/jpucek/BSRH/Data/2025/06/19/'
files = os.listdir(directory)
files = sorted(files)

maskX = 451
maskY = 278
maskR = 47

imageSum = []

delaySlots = []
pulseLength = []

CMx = []
CMy = []

CollGap_H =[]
CollGap_V = []

for nFile in range(110, 111):#len(files)):
    # if 'run_13' in files[nFile]:
    #     print(nFile)

    f = open(directory + files[nFile], 'rb')
    with dctx.stream_reader(f) as reader:
        a = pickle.load(reader)
    print('File n = ', nFile, ' events: ', len(imageSum), ' to ', len(imageSum)+len(a))
    for i in range(len(a)):
        imageSum.append(np.sum(a[i].cam_data['image2D']))
        delaySlots.append(a[i].device_data['GATE-INT']['delaySlots'])
        pulseLength.append(a[i].device_data['GATE-INT']['pulseLength'])
        CMx.append(a[i].device_data['CMASK-X'])
        CMy.append(a[i].device_data['CMASK-Y'])

        CollGap_H.append(a[i].device_data['Collimator-H']['motor_left_upstream']-a[i].device_data['Collimator-H']['motor_right_upstream'])
        CollGap_V.append(a[i].device_data['Collimator-V']['motor_left_upstream'] - a[i].device_data['Collimator-V']['motor_right_upstream'])
    # print(dt.datetime.fromtimestamp(a[0].cam_data['imageTimeStamp'] / 1e9))
    f.close()

#plt.plot(imageSum)

#Verify the mask location on the image
plt.imshow(a[0].cam_data['image2D'])
ax = plt.gca()
circle = plt.Circle((maskX, maskY), maskR, color='r', fill=False)
ax.add_patch(circle)

plt.show()

