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
SIGMA_TO_MM_X = 263.0  # previously hardcoded as 263.0
SIGMA_TO_MM_Y = 361.0

def gaussian(x, amp, cen, wid):#, const):
  return amp*np.exp(-(x-cen)**2/(2*wid**2))#+const

def image_halo(image, occulter, px_size, sigma):
    x1 = 284
    x2 = 684
    y1 = 154
    y2 = 454
    projection_x = np.sum(image[y1:y2, :], axis=0)
    projection_x = projection_x - np.mean(projection_x[:50]) # BTW this is pretty stupid, we should have better way of subtracting the BG

    projection_y = np.sum(image[:, x1:x2], axis=1)
    projection_y = projection_y - np.mean(projection_y[:50])

    correct_measurementX_flag = True
    correct_measurementY_flag = True


    x_temp = []
    x_temp.extend(range(int(maskX - maskR)))
    x_temp.extend(range(int(maskX + maskR), 968))

    try:
        resX, _ = curve_fit(gaussian, x_temp, projection_x[x_temp], (1000, maskX, 15.0))
        if resX[1] > maskX + maskR or resX[1] < maskX - maskR:
            print('Core outside the mask (X)')
            raise Exception('Problem with fit value')
    except:
        resX = [0, maskX, 0]
        correct_measurementX_flag = False

    xl1 = int(resX[1] - sigma * SIGMA_TO_MM_X / px_size[0])
    xr1 = int(resX[1] + sigma * SIGMA_TO_MM_X / px_size[0])

    if (maskX - maskR) < xl1 < (maskX + maskR):
        # This is a problem as the left part is inside the mask and we cannot measure it.
        correct_measurementX_flag = False

    if np.min(projection_x[:xl1]) > 0:
        xl0 = 0
    else:
        i = xl1
        while projection_x[i] > 0 and i > 0:
            i = i - 1
        xl0 = i

    if (maskX - maskR) < xr1 < (maskX + maskR):
        # This is a problem as the left part is inside the mask and we cannot measure it.
        correct_measurementX_flag = False

    if np.min(projection_x[xr1:]) > 0:
        xr0 = 967
    else:
        i = xr1
        while projection_x[i] > 0 and i < 967:
            i = i + 1
        xr0 = i

    if correct_measurementX_flag:
        chargeX = np.sum(projection_x[xl0:xl1]) + np.sum(projection_x[xr1:xr0])
        chargeX = chargeX * cPc * INTENSIFIER_GAIN / exposure

        timeH[bunchIndex].append(BSRH[index].timestamp)
        haloH[bunchIndex].append(chargeX)

    y_temp = []
    y_temp.extend(range(int(maskY - maskR)))  # Here we should be carefull not to put rubbish..
    y_temp.extend(range(int(maskY + maskR), 608))
    # projection_y = projection_y - np.mean(projection_y[:50])

    try:
        resY, _ = curve_fit(gaussian, y_temp, projection_y[y_temp], (1000, maskY, 15.0))  # fit SG to get the assumed center
        if resY[1] > maskY + maskR or resY[1] < maskY - maskR:
            print('Core outside the mask (Y)')
            raise Exception('Problem with fit value')
    except:
        resY = [0, maskY, 0]
        correct_measurementY_flag = False

    # we know the center, so we can make the interval of XX sigma
    yl1 = int(resY[1] - sigma * SIGMA_TO_MM_Y / pixelSizeV)
    yr1 = int(resY[1] + sigma * SIGMA_TO_MM_Y / pixelSizeV)

    if (maskY - maskR) < yl1 < (maskY + maskR):
        # This is a problem as the left part is inside the mask and we cannot measure it.
        correct_measurementY_flag = False

    if np.min(projection_y[:yl1]) > 0:
        yl0 = 0
    else:
        i = yl1
        while projection_y[i] > 0 and i > 0:
            i = i - 1
        yl0 = i

    if (maskY - maskR) < yr1 < (maskY + maskR):
        # This is a problem as the left part is inside the mask and we cannot measure it.
        correct_measurementY_flag = False

    if np.min(projection_y[yr1:]) > 0:
        yr0 = 967
    else:
        i = yr1
        while projection_y[i] > 0 and i < 607:
            i = i + 1
        yr0 = i

    if correct_measurementY_flag:
        chargeY = np.sum(projection_y[yl0:yl1]) + np.sum(projection_y[yr1:yr0])
        chargeY = chargeY * cPc * INTENSIFIER_GAIN / exposure

        timeV[bunchIndex].append(BSRH[index].timestamp)
        haloV[bunchIndex].append(chargeY)

    if bunchIndex + 1 >= count:
        bunchIndex = 0
    else:
        bunchIndex = bunchIndex + 1

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

charge = []

for nFile in range(31,66):#0, len(files)): # Start 250
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

        charge.append(np.sum(a[i].device_data['FBCT1']['averageBunchIntensities'][delaySlots[-1]:delaySlots[-1]+pulseLength[-1]/4]))

        if pulseLength[-1] >= 12000: # Here we want the full beam

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
#
# ## Then we need to get BG (lowest imageSum of the group...)
# BG = []
# for i in range(len(images)):
#     lowestSum = np.sum(images[i][0])
#     BG.append(images[i][0])
#     for j in range(len(images[i])):
#         imSum = np.sum(images[i][j])
#         if imSum < lowestSum:
#             BG[-1] = images[i][j]
#             lowestSum = imSum
#
#
# #plt.ion()
# fig, axs = plt.subplots(2, 2, figsize=(10, 8))
# axs[0,0].imshow(images[1][10]-BG[3])
# axs[0,1].imshow(images[2][10]-BG[3])
# axs[1,0].imshow(images[0][10]-BG[3])
# axs[1,1].imshow(images[3][10]-BG[3])
#
# #Horizontal projections
# span = len(images[2][10][0])/2
# x = np.linspace(-(span/13.15)-1.2, (span/13.15)-1.2, int(2*span))
# plt.figure(2)
# normalisation = np.max(10*np.sum(images[1][10]-BG[3], axis = 0))
#
# plt.plot(x, 10*np.sum(images[1][10]-BG[3], axis = 0)/normalisation, label = 'Imaging')
# plt.plot(x+0.5, 10*np.sum(images[2][10]-BG[3], axis = 0)/normalisation, label = 'LS closed')
# plt.plot(x, np.sum(images[0][10]-BG[3], axis = 0)/normalisation, label = 'LS opened, occulted')
# plt.plot(x, np.sum(images[3][10]-BG[3], axis = 0)/normalisation, label = 'LS closed, occulted')
# plt.vlines(4.15, 0, 0.3, 'red')
# plt.vlines(-4.15, 0, 0.3, 'red')
# plt.legend()
#
# #Vertical projections
# span = len(images[2][10])/2
# x = np.linspace(-(span/20.7)-0.2, (span/20.7)-0.2, int(2*span))
#
# plt.figure(3)
# normalisation = np.max(10*np.sum(images[1][10]-BG[3], axis = 1))
#
# plt.plot(x, 10*np.sum(images[1][10]-BG[3], axis = 1)/normalisation, label = 'Imaging')
# plt.plot(x-0.24, 10*np.sum(images[2][10]-BG[3], axis = 1)/normalisation, label = 'LS closed')
# plt.plot(x, np.sum(images[0][10]-BG[3], axis = 1)/normalisation, label = 'LS opened, occulted')
# plt.plot(x, np.sum(images[3][10]-BG[3], axis = 1)/normalisation, label = 'LS closed, occulted')
# plt.vlines(4.5, 0, 0.3, 'red')
# plt.vlines(-4.5, 0, 0.3, 'red')
# plt.legend()
#
# plt.show()
# plt.plot(imageSum)

####Verify the mask location on the image
# plt.imshow(a[0].cam_data['image2D'])
# ax = plt.gca()
# circle = plt.Circle((maskX, maskY), maskR, color='r', fill=False)
# ax.add_patch(circle)
#
# plt.show()

