import pyjapc
import os
import time
import dill as pickle
import zstandard as zstd
import datetime as dt
import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt

from Acquisition.BWS_logging import BWS_H


def chi2(a, b):
    p = len(a)
    tmp = -1
    if p != len(b):
        print('Cannot calculate chi2 as the lengths of arrays are not the same')
    else:
        tmp = 0
        for i in range(p):
            tmp += ((a[i] - b[i]) ** 2)  # /np.abs(b[i])
            # tmp += (np.abs(a[i]-b[i]))
    return tmp


def gaussian(x, amp, cen, wid):  # , const):
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))  # +const


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def CoolFunc(params, bwsArray, bsrhArray, n=12):  # Params[0] = PSF, Params[1] = pixel size, Params[2] = (scale of bsrh) tail of the PSF, arrays come already as 1000 points from -4 to 4
    newX = np.linspace(-4, 4, 1000)
    crop = 570 ###change here if needed
    tmpPSF = gaussian(newX, 1 / (params[0] * np.sqrt(2 * np.pi)), 0, params[0])
    tmpPSF = tmpPSF / np.sum(tmpPSF)

    # tmpPSF = specGauss(newX, 1, 0, params[0], params[3])
    # tmpPSF = tmpPSF/np.sum(tmpPSF)s
    n = len(bwsArray)
    totalSquares = 0
    for bunch in range(n):
        bsrh = np.interp(newX, params[1] / 1000 * bsrhArray[bunch][0], bsrhArray[bunch][1])
        bws = np.convolve(bwsArray[bunch], tmpPSF, 'same')
        # totalSquares+= chi2(params[2]*bsrh[:crop],bws[:crop])
        totalSquares += chi2(np.sum(bws[:crop]) / np.sum(bsrh[:crop]) * bsrh[:crop], bws[:crop])
    return totalSquares

japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
dctx = zstd.ZstdDecompressor()

DATA_PATH_BWS = '/user/jpucek/BWS/Data'
DATA_PATH_BSRH = '/user/jpucek/BSRH/Data'

bunchN = []

hRatio = 1.008
vRatio = 0.9363

profilesBSRH_H = []
profilesBWS_H = []
profilesBSRH_V = []
profilesBWS_V = []

#Open last file of each BWS:
directoryH = os.path.join(DATA_PATH_BWS+'/Horizontal', time.strftime('%Y/%m/%d/', time.localtime()))
filesH = os.listdir(directoryH)
if filesH == None:
    print('There are no horizontal BWS today...')
else:
    filesH = sorted(filesH)
    # Let's start with horizontal BWS:
    f = open(directoryH + filesH[-1], 'rb')  # Open last file with the latest BWS profile...
    with dctx.stream_reader(f) as reader:
        BWS_H = pickle.load(reader)
    print('BWS_H loaded')

directoryV = os.path.join(DATA_PATH_BWS+'/Vertical', time.strftime('%Y/%m/%d/', time.localtime()))
filesV = os.listdir(directoryV)
if filesV == None:
    print('There are no horizontal BWS today...')
else:
    filesV = sorted(filesV)
    g = open(directoryV + filesV[-1], 'rb')
    with dctx.stream_reader(g) as reader:
        BWS_V = pickle.load(reader)
    print('BWS_V loaded')


directoryBSRH = os.path.join(DATA_PATH_BSRH, time.strftime('%Y/%m/%d/', time.localtime()))
filesBSRH = os.listdir(directoryBSRH)
filesBSRH = sorted(filesBSRH)
##### So we have the BWS data, let's find the corresponding BSRH - we want one image for H and one image for V:

timeH = dt.datetime.fromtimestamp(BWS_H.value['gmtTimestampAcquireIn_ns']/1e9) #Or use 'gmtTimestampAcquireIn_ns'
timeV = dt.datetime.fromtimestamp(BWS_V.value['gmtTimestampAcquireIn_ns']/1e9)
print('BWS Acq Time H: ', timeH)
print('BWS Acq Time V: ', timeV)

count = 1
while True:
    nFiles = len(filesBSRH)
    h = open(directoryBSRH+filesBSRH[nFiles-count], 'rb') #Open last known file
    with dctx.stream_reader(h) as reader:
        BSRH = pickle.load(reader)

    #Now get time of first and last BSRH event:
    tFirstEvent = dt.datetime.fromtimestamp(BSRH[0].cam_data['imageTimeStamp']/1e9)
    tLastEvent = dt.datetime.fromtimestamp(BSRH[-1].cam_data['imageTimeStamp']/1e9)
    print('First event in the BSRH file: ', tFirstEvent)
    print('Last event in the BSRH file:', tLastEvent)

    #So let's see whether the time of BWS is within our file:
    if tFirstEvent < timeH < tLastEvent:
        print('The correct file is loaded!')
        break
    elif tFirstEvent > timeH:
        print('Must load earlier file...')
        count = count + 1 # As long as we don't go below nFiles...
        if count > nFiles: #Whoops, the BSRH file does not exist!
            print('BSRH file count not be found')
            break
    elif tLastEvent < timeH:
        print('The scan took place later than latest file available, waiting for new files')
        while nFiles == len(os.listdir(directoryBSRH)):
            time.sleep(2)
        filesBSRH = os.listdir(directoryBSRH)
        count = 1
##Here we are
if count > nFiles:
    print('Cannot analyse H!')

else:
    # h is the correct file! So let's get the H profiles :) - which ones??
    profilesBSRH_H = []
    if not bunchN:
        bunchN.append(BSRH[0].device_data['GATE-INT']['delaySlots'])
        count = 1
        while bunchN[0] != BSRH[count].device_data['GATE-INT']['delaySlots']:
            bunchN.append(BSRH[count].device_data['GATE-INT']['delaySlots'])
    bunchN = sorted(bunchN)
    start = 0
    for k in range(len(bunchN)):
        bunch = bunchN[k]
        while bunch != BSRH[start].device_data['GATE-INT']['delaySlots'] or start+1 == len(BSRH):
            start = start + 1
        if start == len(BSRH):
            print('Did not find bunch: ', bunchN[k])
            profilesBSRH_H.append([])
        else:
            profilesBSRH_H.append(np.sum(BSRH[start].cam_data['image2D'], axis = 0))

    #Now we want to get the BWS H profiles that correspond to the BSRH ones:
    PointNumber = BWS_H.value['nbPtsPerProjInSet2']
    Pos_in = BWS_H.value['projPositionSet2']
    Proj_in = BWS_H.value['projDataSet2']
    blist = BWS_H.value['bunchList']

    count = int(PointNumber)
    nb = len(Proj_in)
    maxBunch = int(nb / count)

    BWS_H_x = Pos_in[:count] / 1000.0
    newX = np.linspace(-4, 4, 1000)

    for k in range(len(bunchN)):
        bws_index = 0
        while bunchN[k] != blist[bws_index] and bws_index < maxBunch:
            bws_index = bws_index + 1

        if bws_index >= maxBunch:
            print('Could not find bunch: ', bunchN[k])

        else:
            y = Proj_in[k * count:(k + 1) * count]
            y = y - np.mean(y[:50])
            profilesBWS_H.append(moving_average(np.asarray(y), 3))

        #CHECK we have the same number of bunches in BWS and BSRH#
        # We have the BSRH profiles and we have BWS profiles in horizontal...
        if len(profilesBWS_H) == len(profilesBSRH_H):
            print('The number of bunch slots in BWS and BSRH is the same...')

        Hbws = []
        Hbsrh = []
        for bunchIndex in range(len(profilesBWS_H)):
            x = BWS_H_x
            y = profilesBWS_H[bunchIndex]
            out, _ = curve_fit(gaussian, x, y, (1000, 0.5, 0.35))
            x = (x - out[1]) * hRatio
            newY = np.interp(newX, x, y)
            Hbws.append(newY)

        for bunchIndex in range(len(profilesBSRH_H)):
            y = np.asarray(profilesBSRH_H[bunchIndex])
            x = np.arange(len(y))
            out, _ = curve_fit(gaussian, x, y, (60000, 480, 25.0))
            x = x - out[1]
            Hbsrh.append([x, y])

        initGuess = [0.4, 42.5, 0.07, 1.4]
        resultH = minimize(CoolFunc, initGuess, args=(Hbws, Hbsrh), bounds=((0.1, 0.8), (10, 80), (0.005, 2), (0.1, 4)))  # , method = 'L-BFGS-B')
        print(resultH)

        out = resultH.x
        print('PSF = ', out[0])
        print('Pixel size = ', out[1])

        fig, ax = plt.subplots(len(profilesBSRH_H), figsize=(5, 10))

        tmpPSF = gaussian(newX, 1 / (out[0] * np.sqrt(2 * np.pi)), 0, out[0])
        tmpPSF = tmpPSF / np.sum(tmpPSF)

        for bunch in range(len(profilesBSRH_H)):
            bsrh = np.interp(newX, out[1] / 1000 * Hbsrh[bunch][0], Hbsrh[bunch][1])
        bws = np.convolve(Hbws[bunch], tmpPSF, 'same')
        ax[bunch].plot(np.sum(bws) / np.sum(bsrh) * bsrh)
        ax[bunch].plot(bws)


#Now vertical
count = 1
while True:
    nFiles = len(filesBSRH)
    h = open(directoryBSRH + filesBSRH[nFiles - count], 'rb')  # Open last known file
    with dctx.stream_reader(h) as reader:
        BSRH = pickle.load(reader)

    # Now get time of first and last BSRH event:
    tFirstEvent = dt.datetime.fromtimestamp(BSRH[0].cam_data['imageTimeStamp'] / 1e9)
    tLastEvent = dt.datetime.fromtimestamp(BSRH[-1].cam_data['imageTimeStamp'] / 1e9)
    print('First event in the BSRH file: ', tFirstEvent)
    print('Last event in the BSRH file:', tLastEvent)

    # So let's see whether the time of BWS is within our file:
    if tFirstEvent < timeV < tLastEvent:
        print('The correct file is loaded!')
        break
    elif tFirstEvent > timeV:
        print('Must load earlier file...')
        count = count + 1  # As long as we don't go below nFiles...
        if count > nFiles:  # Whoops, the BSRH file does not exist!
            print('BSRH file count not be found')
            break
    elif tLastEvent < timeV:
        print('The scan took place later than latest file available, waiting for new files')
        while nFiles == len(os.listdir(directoryBSRH)):
            time.sleep(2)
        filesBSRH = os.listdir(directoryBSRH)
        count = 1
##Here we are
if count > nFiles:
    print('Cannot analyse V!')

else:
    # h is the correct file! So let's get the H profiles :) - which ones??
    profilesBSRH_V = []
    if not bunchN:
        bunchN.append(BSRH[0].device_data['GATE-INT']['delaySlots'])
        count = 1
        while bunchN[0] != BSRH[count].device_data['GATE-INT']['delaySlots']:
            bunchN.append(BSRH[count].device_data['GATE-INT']['delaySlots'])
    bunchN = sorted(bunchN)
    start = 0
    for k in range(len(bunchN)):
        bunch = bunchN[k]
        while bunch != BSRH[start].device_data['GATE-INT']['delaySlots'] or start + 1 == len(BSRH):
            start = start + 1
        if start == len(BSRH):
            print('Did not find bunch: ', bunchN[k])
            profilesBSRH_V.append([])
        else:
            profilesBSRH_V.append(np.sum(BSRH[start].cam_data['image2D'], axis=1))

    # Now we want to get the BWS H profiles that correspond to the BSRH ones:
    PointNumber = BWS_V.value['nbPtsPerProjInSet2']
    Pos_in = BWS_V.value['projPositionSet2']
    Proj_in = BWS_V.value['projDataSet2']
    blist = BWS_V.value['bunchList']

    count = int(PointNumber)
    nb = len(Proj_in)
    maxBunch = int(nb / count)

    BWS_V_x = Pos_in[:count] / 1000.0
    newX = np.linspace(-4, 4, 1000)

    for k in range(len(bunchN)):
        bws_index = 0
        while bunchN[k] != blist[bws_index] and bws_index < maxBunch:
            bws_index = bws_index + 1

        if bws_index >= maxBunch:
            print('Could not find bunch: ', bunchN[k])

        else:
            y = Proj_in[k * count:(k + 1) * count]
            y = y - np.mean(y[:50])
            profilesBWS_V.append(moving_average(np.asarray(y), 3))

        # CHECK we have the same number of bunches in BWS and BSRH#
        # We have the BSRH profiles and we have BWS profiles in horizontal...
        if len(profilesBWS_H) == len(profilesBSRH_H):
            print('The number of bunch slots in BWS and BSRH is the same...')

        Vbws = []
        Vbsrh = []
        for bunchIndex in range(len(profilesBWS_H)):
            x = BWS_V_x
            y = profilesBWS_V[bunchIndex]
            out, _ = curve_fit(gaussian, x, y, (1000, 0.5, 0.35))
            x = (x - out[1]) * hRatio
            newY = np.interp(newX, x, y)
            Vbws.append(newY)

        for bunchIndex in range(len(profilesBSRH_H)):
            y = np.asarray(profilesBSRH_V[bunchIndex])
            x = np.arange(len(y))
            out, _ = curve_fit(gaussian, x, y, (60000, 480, 25.0))
            x = x - out[1]
            Vbsrh.append([x, y])

        initGuess = [0.4, 42.5, 0.07, 1.4]
        resultV = minimize(CoolFunc, initGuess, args=(Vbws, Vbsrh),bounds=((0.1, 0.8), (10, 80), (0.005, 2), (0.1, 4)))  # , method = 'L-BFGS-B')
        print(resultV)

        out = resultV.x
        print('PSF = ', out[0])
        print('Pixel size = ', out[1])

        fig, ax = plt.subplots(len(profilesBSRH_V), figsize=(5, 10))

        tmpPSF = gaussian(newX, 1 / (out[0] * np.sqrt(2 * np.pi)), 0, out[0])
        tmpPSF = tmpPSF / np.sum(tmpPSF)

        for bunch in range(len(profilesBSRH_H)):
            bsrh = np.interp(newX, out[1] / 1000 * Vbsrh[bunch][0], Vbsrh[bunch][1])
        bws = np.convolve(Vbws[bunch], tmpPSF, 'same')
        ax[bunch].plot(np.sum(bws) / np.sum(bsrh) * bsrh)
        ax[bunch].plot(bws)