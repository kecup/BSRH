import pyjapc
import os
import time
import dill as pickle
import zstandard as zstd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
def gaussian(x, amp, cen, wid):#, const):
  return amp*np.exp(-(x-cen)**2/(2*wid**2))#+const


japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
dctx = zstd.ZstdDecompressor()

DATA_PATH_BSRH = '/user/jpucek/BSRH/Data'
directoryBSRH = os.path.join(DATA_PATH_BSRH, time.strftime('%Y/%m/%d/', time.localtime()))

filesBSRH = os.listdir(directoryBSRH)
filesBSRH = sorted(filesBSRH)
nFiles = len(filesBSRH)

bunchN = []
timeH = [[]]
timeV = [[]]
haloV = [[]]
haloH = [[]]

#Touch this...
maskX = 484 ##Change here the mask position
maskY = 304
maskR = 5
sigma = 3.5 #From where you want to calculate the charge

##Maybe don't touch this unless you know what you are doing...
SIGMA_TO_MM_X = 263.0  # previously hardcoded as 263.0
SIGMA_TO_MM_Y = 361.0
INTENSIFIER_GAIN = 5623
cPc = 55 #This is the charge per count constant, should not need to be changed...
pixelSizeH = 21.0
pixelSizeV = 19.5
x1 = 350
x2 = 650
y1 = 200
y2 = 400

plt.figure(1)
plt.figure(2)
while True:
    filesBSRH = os.listdir(directoryBSRH)
    if nFiles < len(filesBSRH):
        nFiles = len(filesBSRH)
        filesBSRH = sorted(filesBSRH)
        if 'pkl.zst' in filesBSRH[-1]:
            time.sleep(0.2)
            #Hopefully we can perform the analysis before another new file is saved...
            file = open(directoryBSRH + filesBSRH[-1], 'rb')  # Open last known file
            try:
                with dctx.stream_reader(file) as reader:
                    BSRH = pickle.load(reader)
            except:
                nFiles = nFiles-1
                continue

            if not bunchN:#first time running
                bunchN.append(BSRH[0].device_data['GATE-INT']['delaySlots'])
                count = 1
                while bunchN[0] != BSRH[count].device_data['GATE-INT']['delaySlots']:
                    bunchN.append(BSRH[count].device_data['GATE-INT']['delaySlots'])
                    timeH.append([])
                    timeV.append([])
                    haloH.append([])
                    haloV.append([])
                    count = count + 1
                bunchN = sorted(bunchN)

            #Let's find the first bunch slot index:
            firstBunch = BSRH[0].device_data['GATE-INT']['delaySlots']
            count = len(bunchN)
            bunchIndex = bunchN.index(firstBunch)

            print(bunchIndex)
            print(count)

            for index in range(len(BSRH)): #populate the arrays...
                image = BSRH[index].cam_data['image2D']
                projection_x = np.sum(image[y1:y2, :], axis=0)
                projection_x = projection_x - np.mean(projection_x[:50])

                projection_y = np.sum(image[:, x1:x2], axis=1)
                projection_y = projection_y - np.mean(projection_y[:50])

                correct_measurementX_flag = True
                correct_measurementY_flag = True
                exposure = BSRH[index].device_data['GATE-INT']['numberOfPulses']

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

                xl1 = int(resX[1] - sigma * SIGMA_TO_MM_X / pixelSizeH)
                xr1 = int(resX[1] + sigma * SIGMA_TO_MM_X / pixelSizeH)

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
                    resY, _ = curve_fit(gaussian, y_temp, projection_y[y_temp],(1000, maskY, 15.0))  # fit SG to get the assumed center
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
                    bunchIndex = bunchIndex+1

                ##ADD VERTICAL halo charge calculation!!!!
        plt.figure(1)
        plt.clf()
        for k in range(count):
            plt.plot(timeH[k], haloH[k], label = str(bunchN[k]))
        plt.legend()
        plt.draw()
        plt.pause(1.0)

        plt.figure(2)
        plt.clf()
        for k in range(count):
            plt.plot(timeV[k], haloV[k], label=str(bunchN[k]))
        plt.legend()
        plt.draw()
        plt.pause(1.0)



    else:
        plt.pause(1.0)
        #time.sleep(2)