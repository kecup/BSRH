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
directory = '/user/jpucek/BSRH/Data/2025/08/04/'
files = os.listdir(directory)
files = sorted(files)

maskX = 450 ##Change here the mask position
maskY = 277
maskR = 35

imageSum = []

images = [[],[],[],[]] #0 = full image, 1 = occulter in, 2 = LS closed and occulted, 3 = LS closed no occulter
pulses = [[],[],[],[]] # same as above
ndFilters = [[],[],[],[]]


delaySlots = []
pulseLength = []

CMx = []
CMy = []

LS = []

CollGap_H =[]
CollGap_V = []

for nFile in range(0, len(files)): # Start 250
    # if 'run_18' in files[nFile]:
    #     print(nFile)

    f = open(directory + files[nFile], 'rb')
    with dctx.stream_reader(f) as reader:
        a = pickle.load(reader)
    # print('File n = ', nFile, ' events: ', len(imageSum), ' to ', len(imageSum)+len(a))
    for i in range(len(a)):


        imageSum.append(np.sum(a[i].cam_data['image2D']))
        delaySlots.append(a[i].device_data['GATE-INT']['delaySlots'])
        pulseLength.append(a[i].device_data['GATE-INT']['pulseLength'])
        CMx.append(a[i].device_data['CMASK-X'])
        CMy.append(a[i].device_data['CMASK-Y'])

        LS.append(a[i].device_data['LYOT-VU'])
        CollGap_H.append(a[i].device_data['Collimator-H']['motor_left_upstream']-a[i].device_data['Collimator-H']['motor_right_upstream'])
        CollGap_V.append(a[i].device_data['Collimator-V']['motor_left_upstream'] - a[i].device_data['Collimator-V']['motor_right_upstream'])

        if pulseLength[-1] >= 12000:
            #making flags:
            occulter = False
            ls = False # meaning open

            if LS[-1] == 24.6:
                ls = True
            if 62 < CMx[-1] < 64:
                occulter = True # This is the other way around

            filterState = a[i].device_data['FILTERS.B2']['channelState']

            if occulter == False and ls == False: #LS open, occulted image
                images[0].append(a[i].cam_data['image2D'])
                ndFilters[0].append(3*filterState[0]+2*filterState[1]+filterState[2])
                pulses[0].append(a[i].device_data['GATE-INT']['numberOfPulses'])
            elif occulter == True and ls == False:#LS open, full image
                images[1].append(a[i].cam_data['image2D'])
                ndFilters[1].append(3 * filterState[0] + 2 * filterState[1] + filterState[2])
                pulses[1].append(a[i].device_data['GATE-INT']['numberOfPulses'])
            elif occulter == True and ls == True:# LS closed, full image
                images[2].append(a[i].cam_data['image2D'])
                ndFilters[2].append(3 * filterState[0] + 2 * filterState[1] + filterState[2])
                pulses[2].append(a[i].device_data['GATE-INT']['numberOfPulses'])
            elif occulter == False and ls == True:# LS closed, occulted image
                images[3].append(a[i].cam_data['image2D'])
                ndFilters[3].append(3 * filterState[0] + 2 * filterState[1] + filterState[2])
                pulses[3].append(a[i].device_data['GATE-INT']['numberOfPulses'])


    # print(dt.datetime.fromtimestamp(a[0].cam_data['imageTimeStamp'] / 1e9))
    f.close()

## Then we need to get BG (lowest imageSum of the group...)
BG = []
for i in range(len(images)):
    lowestSum = np.sum(images[i][0])
    BG.append(images[i][0])
    for j in range(len(images[i])):
        imSum = np.sum(images[i][j])
        if imSum < lowestSum:
            BG[-1] = images[i][j]
            lowestSum = imSum


#plt.ion()
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0,0].imshow(images[1][10]-BG[3])
axs[0,1].imshow(images[2][10]-BG[3])
axs[1,0].imshow(images[0][10]-BG[3])
axs[1,1].imshow(images[3][10]-BG[3])

#Horizontal projections
span = len(images[2][10][0])/2
x = np.linspace(-(span/13.15)-1.2, (span/13.15)-1.2, int(2*span))
plt.figure(2)
normalisation = np.max(10*np.sum(images[1][10]-BG[3], axis = 0))

plt.plot(x, 10*np.sum(images[1][10]-BG[3], axis = 0)/normalisation, label = 'Imaging')
plt.plot(x+0.5, 10*np.sum(images[2][10]-BG[3], axis = 0)/normalisation, label = 'LS closed')
plt.plot(x, np.sum(images[0][10]-BG[3], axis = 0)/normalisation, label = 'LS opened, occulted')
plt.plot(x, np.sum(images[3][10]-BG[3], axis = 0)/normalisation, label = 'LS closed, occulted')
plt.vlines(4.15, 0, 0.3, 'red')
plt.vlines(-4.15, 0, 0.3, 'red')
plt.legend()

#Vertical projections
span = len(images[2][10])/2
x = np.linspace(-(span/20.7)-0.2, (span/20.7)-0.2, int(2*span))

plt.figure(3)
normalisation = np.max(10*np.sum(images[1][10]-BG[3], axis = 1))

plt.plot(x, 10*np.sum(images[1][10]-BG[3], axis = 1)/normalisation, label = 'Imaging')
plt.plot(x-0.24, 10*np.sum(images[2][10]-BG[3], axis = 1)/normalisation, label = 'LS closed')
plt.plot(x, np.sum(images[0][10]-BG[3], axis = 1)/normalisation, label = 'LS opened, occulted')
plt.plot(x, np.sum(images[3][10]-BG[3], axis = 1)/normalisation, label = 'LS closed, occulted')
plt.vlines(4.5, 0, 0.3, 'red')
plt.vlines(-4.5, 0, 0.3, 'red')
plt.legend()

plt.show()
# plt.plot(imageSum)

####Verify the mask location on the image
# plt.imshow(a[0].cam_data['image2D'])
# ax = plt.gca()
# circle = plt.Circle((maskX, maskY), maskR, color='r', fill=False)
# ax.add_patch(circle)
#
# plt.show()

