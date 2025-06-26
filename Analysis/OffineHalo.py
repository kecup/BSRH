import dill as pickle
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql.functions import hours
from scipy.optimize import curve_fit
import os
import pyjapc
import zstandard as zstd


def gaussian(x, amp, cen, wid):  # , const):
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))  # +const


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

def imageCount(projection, mask, integrationStart,
               xprofile=True):  # projection already after bg subtraction, integrationStart in pixels from center, mask = [x,y,r]
    maskR = mask[2]
    y_temp = []
    if xprofile == True:
        maskPos = mask[0]
        y_temp.extend(range(int(maskPos - maskR)))  # Here we should be carefull not to put rubbish..
        y_temp.extend(range(int(maskPos + maskR), 968))
    else:
        maskPos = mask[1]
        y_temp.extend(range(int(maskPos - maskR)))  # Here we should be carefull not to put rubbish..
        y_temp.extend(range(int(maskPos + maskR), 608))

    correct_measurement_flag = True

    try:
        resY, _ = curve_fit(gaussian, y_temp, projection[y_temp],
                            (1.4e5, maskPos, 40.0))  # fit SG to get the assumed center
        if resY[1] > maskPos + maskR or resY[1] < maskPos - maskR:
            print('Core outside of mask: ', j)
            raise Exception('Problem with fit value')
    except:
        resY = [0, maskPos, 0]
        correct_measurement_flag = False
    yl1 = int(resY[1] - integrationStart)  ## Change here for the norm. sigma
    if yl1 > (maskPos - maskR) and yl1 < (maskPos + maskR):
        # This is a problem as the left part is inside the mask and we cannot measure it.
        print('Halo inside mask: ', j)
        correct_measurement_flag = False

    if np.min(projection[:yl1]) > 0:
        yl0 = 0
    else:
        i = yl1
        while projection[i] > 0 and i > 0:
            i = i - 1
        yl0 = i

    yr1 = int(resY[1] + integrationStart)  ## Change here for the norm. sigma
    if yr1 > (maskPos - maskR) and yr1 < (maskPos + maskR):
        # This is a problem as the left part is inside the mask and we cannot measure it.
        print('Halo inside mask: ', j)
        correct_measurement_flag = False

    if np.min(projection[yr1:]) > 0:
        yr0 = 967  # This works for y profile, should be 607 for x profile
    else:
        i = yr1
        while projection[i] > 0 and i < 607:
            i = i + 1
        yr0 = i
    count = 0
    if correct_measurement_flag == True:  # I just want to append non-zero points so I could get a moving average :)
        count = np.sum(projection[yl0:yl1]) + np.sum(projection[yr1:yr0])

    return count


# Basic setup
directory = '/user/jpucek/BSRH/Data/2025/06/19/'
files = os.listdir(directory)
files = sorted(files)

japc = pyjapc.PyJapc()
dctx = zstd.ZstdDecompressor()
plt.ion()

##IMPORTANT variables:
maskX = 451
maskY = 278
maskR = 45

pixelSizeH = 19.8
pixelSizeV = 19.3
x1 = 284
x2 = 684
y1 = 154
y2 = 454
#m = [MaskX, MaskY, MaskR]

cPc = 27.5  # np.mean(cPc)
sigma = 4.7

SIGMA_TO_MM_X = 263.0  # previously hardcoded as 263.0
SIGMA_TO_MM_Y = 361.0
INTENSIFIER_GAIN = 5623
##Preparing lists for values:

FBCT = [[],[]] #time, values...

haloH = []
haloV = []
timeH = []
timeV = []


numbers = range(110, 162) #RUN13

for j in numbers:
    f = open(directory + files[j], 'rb')
    with dctx.stream_reader(f) as reader:
        BSRH = pickle.load(reader)


    # MaskX = 0
    # MaskY = 0
    # MaskR = 1

    for index in range(len(BSRH)):  # populate the arrays...
        image = BSRH[index].cam_data['image2D']
        time = BSRH[index].timestamp
        time = time.replace(tzinfo = None)
        FBCT[0].append(time)
        intensities = []
        for bunches in range(len(BSRH[index].device_data['FBCT1']['averageBunchIntensities'])):
            value = BSRH[index].device_data['FBCT1']['averageBunchIntensities'][bunches]
            if value > 0:
                intensities.append(value)

        FBCT[1].append(intensities)
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

            timeH.append(BSRH[index].timestamp)
            haloH.append(chargeX)

        y_temp = []
        y_temp.extend(range(int(maskY - maskR)))  # Here we should be carefull not to put rubbish..
        y_temp.extend(range(int(maskY + maskR), 608))
        # projection_y = projection_y - np.mean(projection_y[:50])

        try:
            resY, _ = curve_fit(gaussian, y_temp, projection_y[y_temp],
                                (1000, maskY, 15.0))  # fit SG to get the assumed center
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

            timeV.append(time)
            haloV.append(chargeY)

## Here we have to perform the BWS analysis:

def haloBWS(directory, start, stop, sigma, size, bStart, bStop): #Calculates the halo from the BWS from file start to stop, the halo from sigma to infty, using the beamsize = size and summing bunches bStart to bStop
    files = os.listdir(directory)
    files = sorted(files)

    time = []
    halo = np.zeros(int(stop-start))

    event = 0
    for nFile in range(start, stop):
        f = open(directory + files[nFile], 'rb')
        with dctx.stream_reader(f) as reader:
            a = pickle.load(reader)

        time.append(dt.datetime.fromtimestamp(a.value['gmtTimestampAcquireIn_ns'] / 1e9))
        PointNumber = a.value['nbPtsPerProjInSet2']
        Pos_in = a.value['projPositionSet2']
        Proj_in = a.value['projDataSet2']
        blist = a.value['bunchList']

        count = int(PointNumber)
        nb = len(Proj_in)
        maxBunch = int(nb / count)

        x = Pos_in[:count] / 1000.0
        newX = np.linspace(-10, 10, 1000) #This is sufficient for the analysis...

        #profiles = []
        for k in range(bStart, bStop):
            y = Proj_in[k * count:(k + 1) * count]
            y = y - np.mean(y[:50])
            y= moving_average(np.asarray(y), 1) #MOVING AVERAGE HERE, change for probably better scaling of the PSF output??
            out, _ = curve_fit(gaussian, x, y, (1000, 0.3, 0.28))
            x = (x - out[1]) # Recenter...
            newY = np.interp(newX, x, y)

            highLeftPosition = -sigma*size
            highRightPosition = sigma*size

            lh = find_nearest_index(newX, highLeftPosition)
            rh = find_nearest_index(newX, highRightPosition)

            if newY[lh] < 0 or newY[rh] < 0:
                ## This means no charge measurement -> so nothing:
                print('No charge found in scan:', event, ' bunch:', k)
            else:
                tmp = lh
                while newY[tmp] > 0 and tmp > 0:  # Getting the last positive value of the scan
                    tmp = tmp - 1
                ll = tmp + 1

                tmp = rh
                while newY[tmp] > 0 and tmp < len(newY):
                    tmp = tmp + 1
                rl = tmp - 1
                ## Now calculate halo - we need FBCT normalisation.... let's find the closest one...
                time = dt.datetime.fromisoformat(a.value['acqTime'])
                time = time - dt.timedelta(hours=2)
                fbctIndex = find_nearest_index(FBCT[0], time)#dt.datetime.fromisoformat(a.value['acqTime']))
                fbctCharge = FBCT[1][fbctIndex][k]
                normalisation = fbctCharge/np.sum(newY[ll:rl])
                halo[event] += (np.sum(newY[ll:lh]) + np.sum(newY[rh:rl]))*normalisation
        event += 1
    return time, halo

directoryV = '/user/jpucek/BWS/Data/Vertical/2025/06/19/'
directoryH = '/user/jpucek/BWS/Data/Horizontal/2025/06/19/'

# BWS for cycle:

bwsTime, bwsHalo = haloBWS(directoryV, 65, 88, 4.7, 0.386, 4, 14)


plt.plot(timeV, haloV)
plt.plot(bwsTime, bwsHalo)
plt.show()

# plt.figure(1)
# plt.imshow(image)
# ax = plt.gca()
# MaskX = 486
# MaskY = 308
# MaskR = 47
# circle = plt.Circle((MaskX, MaskY), MaskR, color='r', fill=False)
# ax.add_patch(circle)
# plt.show()
