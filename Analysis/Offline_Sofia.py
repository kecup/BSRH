from datetime import timedelta

import pyjapc
import os
import time
import dill as pickle
import zstandard as zstd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from future.backports.datetime import datetime
from scipy.optimize import curve_fit


def gaussian(x, amp, cen, wid):#, const):
  return amp*np.exp(-(x-cen)**2/(2*wid**2))#+const

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_run_data(directory: str, files, run: str):
    # data = []
    # for file in files:
    #     if run in file:
    #         with open(directory + file, 'rb') as f:
    #             with dctx.stream_reader(f) as reader:
    #                 file_data = pickle.load(reader)
    #                 data.append(file_data)
    # return data
    run_files = [os.path.join(directory, f) for f in files if run in f]
    run_files = sorted(run_files)
    return run_files

def haloBWS(x_out, y_out, sigma, size): #Calculates the halo from the BWS from file start to stop, the halo from sigma to infty, using the beamsize = size and summing bunches bStart to bStop

    x_in = x_out
    x_in = x_in[1:-1]  # Due to moving average

    newX = np.linspace(-10, 10, 2000) #This is sufficient for the analysis...

    #profiles = []
    y = y_out - np.mean(y_out[:50])
    y= moving_average(np.asarray(y), 3) #MOVING AVERAGE HERE, change for probably better scaling of the PSF output??

    out, _ = curve_fit(gaussian, x_in, y, (1000, 0.3, 0.28))
    x_found = (x_in - out[1]) # Recenter...
    newY = np.interp(newX, x_found, y)

    #plt.plot(newX, newY)
    #plt.show()

    highLeftPosition = -sigma*size
    highRightPosition = sigma*size

    lh = find_nearest_index(newX, highLeftPosition)
    rh = find_nearest_index(newX, highRightPosition)
    halo = -1

    if newY[lh] < 0 or newY[rh] < 0:
        ## This means no charge measurement -> so nothing:
        print('No charge found in scan...')
    else:
        tmp = lh
        while newY[tmp] > 0 and tmp > 0:  # Getting the last positive value of the scan
            tmp = tmp - 1
        ll = tmp + 1

        tmp = rh
        while newY[tmp] > 0 and tmp < len(newY)-1:
            tmp = tmp + 1
        rl = tmp - 1

        halo = (np.sum(newY[ll:lh]) + np.sum(newY[rh:rl]))/np.sum(newY[ll:rl])
    return halo #This halo just needs to be multiplied by the FBCT and we get the charge...

japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
dctx = zstd.ZstdDecompressor()

#DATA_PATH_BSRH = '/user/jpucek/BSRH/Data'
directoryBSRH = '/user/jpucek/BSRH/Data/2025/06/20/' #os.path.join(DATA_PATH_BSRH, time.strftime('%Y/%m/%d/', time.localtime()))

fileBWSh = '/user/jpucek/GitLab/bsrh-scripts/Analysis/BWS_Horizontal_Sofia_out.pkl'
fileBWSv = '/user/jpucek/GitLab/bsrh-scripts/Analysis/BWS_Vertical_Sofia_out.pkl'

H_BWS_time = [[],[],[],[],[],[],[]]
H_BWS_halo = [[],[],[],[],[],[],[]]

V_BWS_time = [[],[],[],[],[],[],[]]
V_BWS_halo = [[],[],[],[],[],[],[]]

f = open(fileBWSh,'rb')
g = open(fileBWSv,'rb')

V = pickle.load(f)
vindex = 23
H = pickle.load(g)
hindex = 20

hSize = 257.0/1000.0
vSize = 386.0/1000.0

f.close()
g.close()

filesBSRH = os.listdir(directoryBSRH)
filesBSRH = sorted(filesBSRH)
nFiles = len(filesBSRH)

runs = ['run_03', 'run_07', 'run_10', 'run_12', 'run_14']

#Touch this...
maskX = 455 ##Change here the mask position
maskY = 300
maskR = 40
sigma = 3.5 #From where you want to calculate the charge

##Maybe don't touch this unless you know what you are doing...
SIGMA_TO_MM_X = 263.0  # previously hardcoded as 263.0
SIGMA_TO_MM_Y = 361.0

INTENSIFIER_GAIN = 5623
cPc = 43.5 #This is the charge per count constant, should not need to be changed...

pixelSizeH = 19.8
pixelSizeV = 17.5

x1 = 284
x2 = 684
y1 = 154
y2 = 454

#plt.figure(1)
#plt.figure(2)

filesBSRH = os.listdir(directoryBSRH)
results = {}
cmask_x = []
cmask_y = []

for run in runs:

    print("Analysing " + run)
    run_data = get_run_data(directoryBSRH, filesBSRH, run)

    for file_name in run_data:
        with open(file_name, 'rb') as f:
            with dctx.stream_reader(f) as reader:
                file_data = pickle.load(reader)

        for event in file_data:
            image = event.cam_data['image2D']
            projection_x = np.sum(image[y1:y2, :], axis=0)
            projection_x = projection_x - np.mean(projection_x[:50])

            projection_y = np.sum(image[:, x1:x2], axis=1)
            projection_y = projection_y - np.mean(projection_y[:50])

            correct_measurementX_flag = True
            correct_measurementY_flag = True

            cmask_x.append(event.device_data['CMASK-X'])
            cmask_y.append(event.device_data['CMASK-Y'])

            exposure = event.device_data['GATE-INT']['numberOfPulses']
            delaySlots = int(event.device_data["GATE-INT"]["delaySlots"])

            tmpFBCT = event.device_data['FBCT1']['averageBunchIntensities']
            tmpFBCT = [x for x in tmpFBCT if x != 0]

            htime = H.time[hindex] - dt.timedelta(hours=2)
            htime = htime.replace(tzinfo = dt.timezone.utc)

            vtime = V.time[vindex] - dt.timedelta(hours=2)
            vtime = vtime.replace(tzinfo = dt.timezone.utc)

            if abs((event.timestamp - htime).total_seconds()) < 0.250:
                # this is where we should perform the BWS analysis using the FBCT data...
                for bunch_index in range(len(H.x[hindex])):
                #So there are 7 bunches, how to easily save this? Into a list...
                    halo = haloBWS(H.x[hindex][bunch_index], H.y[hindex][bunch_index], sigma, hSize)*tmpFBCT[bunch_index]
                    H_BWS_time[bunch_index].append(event.timestamp)
                    H_BWS_halo[bunch_index].append(halo)
                #Now look at the next BWS
                hindex+=1
            elif (event.timestamp - htime).total_seconds() > 0:
                hindex += 1
                #print(event.timestamp, htime)

            if abs((event.timestamp - vtime).total_seconds()) < 0.250:
                # this is where we should perform the BWS analysis using the FBCT data...
                for bunch_index in range(len(V.x[hindex])):
                #So there are 7 bunches, how to easily save this? Into a list...
                    halo = haloBWS(V.x[vindex][bunch_index], V.y[vindex][bunch_index], sigma, vSize)*tmpFBCT[bunch_index]
                    V_BWS_time[bunch_index].append(event.timestamp)
                    V_BWS_halo[bunch_index].append(halo)
                #Now look at the next BWS
                vindex+=1

            elif(event.timestamp - vtime).total_seconds() > 0:
                vindex+=1

            if delaySlots not in results:
                results[delaySlots] = {'timeH': [], 'haloH': [], 'timeV': [], 'haloV': []}

            x_temp = []
            x_temp.extend(range(int(maskX - maskR)))
            x_temp.extend(range(int(maskX + maskR), 968))

            try:
                resX, _ = curve_fit(gaussian, x_temp, projection_x[x_temp], (45000, maskX, 20.0))
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
                fArr = event.device_data['FILTERS.B2']['channelState']
                in_filter = pow(12.6, 3 * fArr[0] + 2 * fArr[1] + fArr[2])
                chargeX = chargeX * cPc * INTENSIFIER_GAIN * in_filter / exposure

                results[delaySlots]['timeH'].append(event.timestamp)
                results[delaySlots]['haloH'].append(chargeX)

            y_temp = []
            y_temp.extend(range(int(maskY - maskR)))  # Here we should be carefull not to put rubbish..
            y_temp.extend(range(int(maskY + maskR), 608))
            # projection_y = projection_y - np.mean(projection_y[:50])

            try:
                resY, _ = curve_fit(gaussian, y_temp, projection_y[y_temp],(45000, maskY, 20.0))  # fit SG to get the assumed center
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
                fArr = event.device_data['FILTERS.B2']['channelState']
                in_filter = pow(12.6, 3 * fArr[0] + 2 * fArr[1] + fArr[2])
                chargeY = chargeY * cPc * INTENSIFIER_GAIN * in_filter / exposure

                results[delaySlots]['timeV'].append(event.timestamp)
                results[delaySlots]['haloV'].append(chargeY)
                ##ADD VERTICAL halo charge calculation!!!!

OutFile = open('HaloCharge_v2.pkl', 'wb')
pickle.dump(results, OutFile)
OutFile.close()

BWS_data = [H_BWS_time, H_BWS_halo, V_BWS_time, V_BWS_halo]

BWS_out = open('BWSCharge_v1.pkl', 'wb')
pickle.dump(BWS_data, BWS_out)
BWS_out.close()