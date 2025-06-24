import pyjapc
import os
import time
import dill as pickle
import zstandard as zstd
import datetime as dt
import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt

from scipy import stats
from skimage import restoration

File_H = 47
File_V = 51
File_BSRH = 54

def chi2(a, b):
    p = len(a)
    tmp = -1
    if p != len(b):
        print('Cannot calculate chi2 as the lengths of arrays are not the same')
    else:
        tmp = 0
        for i in range(p):
            tmp += ((a[i] - b[i]) ** 2)  #/np.abs(b[i])
            # tmp += (np.abs(a[i]-b[i]))
    return tmp/p


def gaussian(x, amp, cen, wid):  # , const):
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))  # +const


def specGauss(x, amp, cen, wid, kh):
    arg = (x - cen) / wid
    value = np.where(arg <= kh,
                     amp * np.exp(-0.5 * arg ** 2),
                     amp * np.exp(0.5 * kh ** 2 - kh * arg))
    return value


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] -= ret[:-n]
    result = ret[n - 1:] / n
    pad = [result[0]] * (n - 1)
    return np.array(pad + list(result))


def CoolFuncH(params, bwsArray, bsrhArray, fbctArray, n=12):  # Params[0] = PSF, Params[1] = pixel size, Params[2] = (scale of bsrh) tail of the PSF, arrays come already as 1000 points from -4 to 4
    newX = np.linspace(-4, 4, 1000)
    crop = 570#600 ###change here if needed
    tmpPSF = gaussian(newX, 1 / (params[0] * np.sqrt(2 * np.pi)), 0, params[0])
    tmpPSF = tmpPSF / np.sum(tmpPSF)

    # tmpPSF = specGauss(newX, 1, 0, params[0], params[3])
    # tmpPSF = tmpPSF/np.sum(tmpPSF)
    n = len(bwsArray)
    totalSquares = 0
    for bunch in range(n):
        bsrh = np.interp(newX, params[1] / 1000 * bsrhArray[bunch][0], bsrhArray[bunch][1])
        #bsrh = restoration.richardson_lucy(bsrh, tmpPSF, 5,False)
        bsrh = fbctArray[bunch]/np.sum(bsrh)*bsrh

        bws = np.convolve(bwsArray[bunch], tmpPSF, 'same')
        bws = fbctArray[bunch]/np.sum(bwsArray[bunch])*bwsArray[bunch]

        # totalSquares+= chi2(params[2]*bsrh[:crop],bws[:crop])
        totalSquares += chi2(bsrh[:crop], bws[:crop])
    return totalSquares

def CoolFuncV(params, bwsArray, bsrhArray, fbctArray, n=12):  # Params[0] = PSF, Params[1] = pixel size, Params[2] = (scale of bsrh) tail of the PSF, arrays come already as 1000 points from -4 to 4
    newX = np.linspace(-4, 4, 1000)
    crop = -1 ###change here if needed
    tmpPSF = gaussian(newX, 1 / (params[0] * np.sqrt(2 * np.pi)), 0, params[0])
    tmpPSF = tmpPSF / np.sum(tmpPSF)

    # tmpPSF = specGauss(newX, 1, 0, params[0], params[3])
    # tmpPSF = tmpPSF/np.sum(tmpPSF)s
    n = len(bwsArray)
    totalSquares = 0
    for bunch in range(n):
        bsrh = np.interp(newX, params[1] / 1000 * bsrhArray[bunch][0], bsrhArray[bunch][1])
        # bsrh = restoration.richardson_lucy(bsrh, tmpPSF, 5, False)
        bsrh = fbctArray[bunch] / np.sum(bsrh) * bsrh

        bws = np.convolve(bwsArray[bunch], tmpPSF, 'same')
        bws = fbctArray[bunch]/np.sum(bws)*bws

        # totalSquares+= chi2(params[2]*bsrh[:crop],bws[:crop])
        totalSquares += chi2(bsrh[:crop], bws[:crop])
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

results = {}

#Open last file of each BWS:
directoryH = os.path.join(DATA_PATH_BWS+'/Horizontal/2025/06/19/')#, time.strftime('%Y/%m/%d/', time.localtime()))
filesH = os.listdir(directoryH)
if not filesH:
    print('There are no horizontal BWS today...')
else:
    filesH = sorted(filesH)
    # Let's start with horizontal BWS:
    #Change to file 23
    f = open(directoryH + filesH[File_H], 'rb')  # Open last file with the latest BWS profile...
    with dctx.stream_reader(f) as reader:
        BWS_H = pickle.load(reader)
    print('BWS_H loaded')

directoryV = os.path.join(DATA_PATH_BWS+'/Vertical/2025/06/19/')#, time.strftime('%Y/%m/%d/', time.localtime()))
filesV = os.listdir(directoryV)
if not filesV:
    print('There are no vertical BWS today...')
else:
    filesV = sorted(filesV)
    g = open(directoryV + filesV[File_V], 'rb')
    with dctx.stream_reader(g) as reader:
        BWS_V = pickle.load(reader)
    print('BWS_V loaded')


directoryBSRH = os.path.join(DATA_PATH_BSRH, '2025/06/19/')#time.strftime('%Y/%m/%d/', time.localtime()))
filesBSRH = os.listdir(directoryBSRH)
filesBSRH = sorted(filesBSRH)
##### So we have the BWS data, let's find the corresponding BSRH - we want one image for H and one image for V:

timeH = dt.datetime.fromtimestamp(BWS_H.value['gmtTimestampAcquireIn_ns']/1e9) #Or use 'gmtTimestampAcquireIn_ns'
timeV = dt.datetime.fromtimestamp(BWS_V.value['gmtTimestampAcquireIn_ns']/1e9)
print('BWS Acq Time H: ', timeH)
print('BWS Acq Time V: ', timeV)

count = len(filesBSRH) - File_BSRH
while True:
    nFiles = len(filesBSRH)
    h = open(directoryBSRH+filesBSRH[nFiles-count], 'rb') #Open last known file
    with dctx.stream_reader(h) as reader:
        BSRH = pickle.load(reader)

    #Now get time of first and last BSRH event:
    tFirstEvent = dt.datetime.fromtimestamp(BSRH[0].cam_data['imageTimeStamp']/1e9)#+dt.timedelta(hours=2)
    tLastEvent = dt.datetime.fromtimestamp(BSRH[-1].cam_data['imageTimeStamp']/1e9)#+dt.timedelta(hours=2)

    #print('First event in the BSRH file: ', tFirstEvent)
    #print('Last event in the BSRH file:', tLastEvent)

    #So let's see whether the time of BWS is within our file:
    if tFirstEvent < timeH < tLastEvent:
        print('The correct file is loaded!')
        break
    elif tFirstEvent > timeH:
        #print('Must load earlier file...')
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
    fbct_H = []
    if not bunchN:
        bunchN.append(BSRH[0].device_data['GATE-INT']['delaySlots'])
        count = 1
        while bunchN[0] != BSRH[count].device_data['GATE-INT']['delaySlots']:
            bunchN.append(BSRH[count].device_data['GATE-INT']['delaySlots'])
            count += 1
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
            prof = np.sum(BSRH[start].cam_data['image2D'][154:454,:], axis = 0)
            prof = prof - np.mean(prof[:50])
            profilesBSRH_H.append(prof)
            bStart = bunch
            bStop = bunch + int(BSRH[start].device_data['GATE-INT']['numberOfPulses']/4.0)
            fbct_H.append(np.sum(BSRH[start].device_data['FBCT1']['averageBunchIntensities'][bStart:bStop]))

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
            profilesBWS_H.append(moving_average(np.asarray(y), 3)) #MOVING AVERAGE HERE, change for probably better scaling of the PSF output??

    #CHECK we have the same number of bunches in BWS and BSRH#
    # We have the BSRH profiles and we have BWS profiles in horizontal...
    if len(profilesBWS_H) == len(profilesBSRH_H):
        print('The number of bunch slots in BWS and BSRH is the same...')

    Hbws = []
    Hbsrh = []
    for bunchIndex in range(len(profilesBWS_H)):
        x = BWS_H_x
        y = profilesBWS_H[bunchIndex]
        out, _ = curve_fit(gaussian, x, y, (1000, 0.3, 0.28))
        x = (x - out[1]) * hRatio
        newY = np.interp(newX, x, y)
        Hbws.append(newY)

    for bunchIndex in range(len(profilesBSRH_H)):
        y = np.asarray(profilesBSRH_H[bunchIndex])
        x = np.arange(len(y))
        out, _ = curve_fit(gaussian, x, y, (170000, 511, 25.0))
        x = x - out[1]
        Hbsrh.append([x, y])


    ## Here we have our lists prepared, let's make first guess :)
    sigmaBSRH_H = []
    sigmaBWS_H = []

    initBWS = [1000, 0.0, 0.03]
    initBSRH = [170000, 0.0, 25.0]
    for bindex in range(len(Hbws)):
        res, _ = curve_fit(gaussian, newX, Hbws[bindex], initBWS) # This has physical units from -4 to 4 mm
        sigmaBWS_H.append(res[2])

        res, _ = curve_fit(gaussian, Hbsrh[bindex][0][:570], Hbsrh[bindex][1][:570]) # This has not physical units so px...
        sigmaBSRH_H.append(res[2])

    slope, intercept, r, p, std_err = stats.linregress(np.asarray(sigmaBWS_H)**2, np.asarray(sigmaBSRH_H)**2)
    print('Magnification = ', np.sqrt(1/slope)*1e3, 'um/px, PSF = ', np.sqrt(intercept/slope)*1e3, 'um')

    plt.figure(11)
    plt.plot(np.asarray(sigmaBWS_H) ** 2, np.asarray(sigmaBSRH_H) ** 2, 'o')

    initGuess = [np.sqrt(intercept/slope), np.sqrt(1/slope)*1e3, 1.0, 1.4]
    resultH = minimize(CoolFuncH, initGuess, args=(Hbws, Hbsrh, fbct_H), bounds=((0.1, 0.8), (15, 45), (0.1, 2.0), (0.1, 4))) #, method = 'L-BFGS-B')
    #print(resultH)

    out = resultH.x
    results["horizontal"] = (out[0], out[1])
    print('PSF = ', np.round(out[0],3))
    print('Pixel size = ', np.round(out[1],2))
    print('Proportionality factor = ', np.round(out[2],5))

    plotList = [0, 1, 10, 11, 24, 25]
    fig, ax = plt.subplots(len(plotList), figsize=(5, 10))
    fig.suptitle("Horizontal Comparison of BSRH and BWS Profiles")

    tmpPSF = gaussian(newX, 1 / (out[0] * np.sqrt(2 * np.pi)), 0, out[0])
    tmpPSF = tmpPSF / np.sum(tmpPSF)

    for slot, bunch in enumerate(plotList):
        bsrh = np.interp(newX, out[1] / 1000 * Hbsrh[bunch][0], Hbsrh[bunch][1])
        bsrh = fbct_H[bunch] / np.sum(bsrh) * bsrh
        bws = np.convolve(Hbws[bunch], tmpPSF, 'same')
        bws = fbct_H[bunch] / np.sum(bws) * bws
        ax[slot].plot(bsrh, label="BSRH")
        ax[slot].plot(bws, label="BWS")
        ax[slot].legend()
        # ax[bunch].set_title("Bunch: " + str(bunch))

    # plt.show()

#Now vertical
count = len(filesBSRH) - File_BSRH
while True:
    nFiles = len(filesBSRH)
    h = open(directoryBSRH + filesBSRH[nFiles - count], 'rb')  # Open last known file
    with dctx.stream_reader(h) as reader:
        BSRH = pickle.load(reader)

    # Now get time of first and last BSRH event:
    tFirstEvent = dt.datetime.fromtimestamp(BSRH[0].cam_data['imageTimeStamp'] / 1e9)
    tLastEvent = dt.datetime.fromtimestamp(BSRH[-1].cam_data['imageTimeStamp'] / 1e9)
    #print('First event in the BSRH file: ', tFirstEvent)
    #print('Last event in the BSRH file:', tLastEvent)

    # So let's see whether the time of BWS is within our file:
    if tFirstEvent < timeV < tLastEvent:
        print('The correct file is loaded!')
        break
    elif tFirstEvent > timeV:
        #print('Must load earlier file...')
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
    fbct_V = []
    if not bunchN:
        bunchN.append(BSRH[0].device_data['GATE-INT']['delaySlots'])
        count = 1
        while bunchN[0] != BSRH[count].device_data['GATE-INT']['delaySlots']:
            bunchN.append(BSRH[count].device_data['GATE-INT']['delaySlots'])
            count += 1
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
            prof = np.sum(BSRH[start].cam_data['image2D'][:,284:684], axis=1)
            prof = prof - np.mean(prof[:50])
            profilesBSRH_V.append(prof)
            bStart = bunch
            bStop = bunch + int(BSRH[start].device_data['GATE-INT']['numberOfPulses'] / 4.0)
            fbct_V.append(np.sum(BSRH[start].device_data['FBCT1']['averageBunchIntensities'][bStart:bStop]))

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
            profilesBWS_V.append(moving_average(np.asarray(y), 1))  #MOVING AVERAGE HERE, change for probably better scaling of the PSF output??

    # CHECK we have the same number of bunches in BWS and BSRH#
    # We have the BSRH profiles and we have BWS profiles in vertical...
    if len(profilesBWS_V) == len(profilesBSRH_V):
        print('The number of bunch slots in BWS and BSRH is the same...')

    Vbws = []
    Vbsrh = []
    for bunchIndex in range(len(profilesBWS_V)):
        x = BWS_V_x
        y = profilesBWS_V[bunchIndex]
        out, _ = curve_fit(gaussian, x, y, (1000, 0.5, 0.35))
        x = (x - out[1]) * hRatio
        newY = np.interp(newX, x, y)
        Vbws.append(newY)

    for bunchIndex in range(len(profilesBSRH_V)):
        y = np.asarray(profilesBSRH_V[bunchIndex])
        x = np.arange(len(y))
        out, _ = curve_fit(gaussian, x, y, (60000, 480, 25.0))
        x = x - out[1]
        Vbsrh.append([x, y])

    sigmaBWS_V = []
    sigmaBSRH_V = []
    for bindex in range(len(Hbws)):
        res, _ = curve_fit(gaussian, newX, Vbws[bindex], initBWS) # This has physical units from -4 to 4 mm
        sigmaBWS_V.append(res[2])

        res, _ = curve_fit(gaussian, Vbsrh[bindex][0], Vbsrh[bindex][1]) # This has not physical units so px...
        sigmaBSRH_V.append(res[2])

    slope, intercept, r, p, std_err = stats.linregress(np.asarray(sigmaBWS_V)**2, np.asarray(sigmaBSRH_V)**2)
    print('Magnification = ', np.sqrt(1/slope)*1e3, 'um/px, PSF = ', np.sqrt(intercept/slope)*1e3, 'um')

    plt.figure(13)
    plt.plot(np.asarray(sigmaBWS_V)**2, np.asarray(sigmaBSRH_V)**2, 'o')

    initGuess = [np.sqrt(intercept/slope)*1e3, np.sqrt(1/slope)*1e3, 1, 1.4]
    resultV = minimize(CoolFuncV, initGuess, args=(Vbws, Vbsrh, fbct_V),bounds=((0.1, 0.5), (10, 22), (0.1, 2.0), (0.1, 4)))  # , method = 'L-BFGS-B')
    #print(resultV)

    out = resultV.x
    results["vertical"] = (out[0], out[1])
    print('PSF = ', np.round(out[0],3))
    print('Pixel size = ', np.round(out[1],2))
    print('Proportionality factor = ', np.round(out[2],5))

    plotList = [0,1,10,11,24,25]
    fig, ax = plt.subplots(len(plotList), figsize=(5, 10))
    fig.suptitle("Vertical Comparison of BSRH and BWS Profiles")

    tmpPSF = gaussian(newX, 1 / (out[0] * np.sqrt(2 * np.pi)), 0, out[0])
    tmpPSF = tmpPSF / np.sum(tmpPSF)

    for slot, bunch in enumerate(plotList):
        bsrh = np.interp(newX, out[1] / 1000 * Vbsrh[bunch][0], Vbsrh[bunch][1])
        bsrh = fbct_V[bunch]/np.sum(bsrh)*bsrh
        bws = np.convolve(Vbws[bunch], tmpPSF, 'same')
        bws = fbct_V[bunch]/np.sum(bws)*bws
        ax[slot].plot(bsrh, label="BSRH")
        ax[slot].plot(bws, label="BWS")
        ax[slot].legend()
        # ax[bunch].set_title("Bunch: " + str(bunch))

    print("Results: (PSF, Pixel Size)")
    print(results)
    plt.show()