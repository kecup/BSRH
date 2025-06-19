import os
import time
import dill as pickle
import zstandard as zstd
import datetime as dt
import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
import pyjapc

def chi2(a, b):
    if len(a) != len(b):
        print('Cannot calculate chi2 as lengths differ')
        return -1
    return np.sum((a - b) ** 2)

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] -= ret[:-n]
    result = ret[n - 1:] / n
    pad = [result[0]] * (n - 1)
    return np.array(pad + list(result))

def CoolFunc(params, bwsArray, bsrhArray, crop):
    newX = np.linspace(-4, 4, 1000)
    tmpPSF = gaussian(newX, 1 / (params[0] * np.sqrt(2 * np.pi)), 0, params[0])
    tmpPSF /= np.sum(tmpPSF)

    totalSquares = 0
    for bunch in range(len(bwsArray)):
        bsrh = np.interp(newX, params[1] / 1000 * bsrhArray[bunch][0], bsrhArray[bunch][1])
        bws = np.convolve(bwsArray[bunch], tmpPSF, 'same')
        totalSquares += chi2(np.sum(bws[:crop]) / np.sum(bsrh[:crop]) * bsrh[:crop], bws[:crop])
    return totalSquares

def load_BWS_file(path, dctx):
    files = sorted(os.listdir(path))
    if not files:
        print(f"No BWS files in {path}")
        return None
    with dctx.stream_reader(open(path + files[-1], 'rb')) as reader:
        return pickle.load(reader)

def find_matching_BSRH(directoryBSRH, BWS_time, dctx):
    filesBSRH = sorted(os.listdir(directoryBSRH))
    count = 1
    while True:
        nFiles = len(filesBSRH)
        with dctx.stream_reader(open(directoryBSRH + filesBSRH[nFiles - count], 'rb')) as reader:
            BSRH = pickle.load(reader)
        tFirst = dt.datetime.fromtimestamp(BSRH[0].cam_data['imageTimeStamp'] / 1e9)
        tLast = dt.datetime.fromtimestamp(BSRH[-1].cam_data['imageTimeStamp'] / 1e9)
        if tFirst < BWS_time < tLast:
            print('Matching BSRH file found.')
            return BSRH
        elif tFirst > BWS_time:
            count += 1
            if count > nFiles:
                print("No matching BSRH file found.")
                return None
        elif tLast < BWS_time:
            print("Waiting for new BSRH files...")
            while nFiles == len(os.listdir(directoryBSRH)):
                time.sleep(2)
            filesBSRH = sorted(os.listdir(directoryBSRH))
            count = 1

def process_profiles(BSRH, BWS, direction, bunchN, ratio):
    profilesBSRH = []
    profilesBWS = []

    axis = 0 if direction == 'H' else 1
    PointNumber = BWS.value['nbPtsPerProjInSet2']
    Pos_in = BWS.value['projPositionSet2']
    Proj_in = BWS.value['projDataSet2']
    blist = BWS.value['bunchList']
    count = int(PointNumber)
    nb = len(Proj_in)
    maxBunch = int(nb / count)
    BWS_x = Pos_in[:count] / 1000.0
    newX = np.linspace(-4, 4, 1000)

    # BSRH profiles
    if not bunchN:
        bunchN.append(BSRH[0].device_data['GATE-INT']['delaySlots'])
        idx = 1
        while bunchN[0] != BSRH[idx].device_data['GATE-INT']['delaySlots']:
            bunchN.append(BSRH[idx].device_data['GATE-INT']['delaySlots'])
            idx += 1
    bunchN = sorted(bunchN)

    start = 0
    for bunch in bunchN:
        while bunch != BSRH[start].device_data['GATE-INT']['delaySlots'] or start + 1 == len(BSRH):
            start += 1
        if start == len(BSRH):
            profilesBSRH.append([])
        else:
            prof = np.sum(BSRH[start].cam_data['image2D'], axis=axis)
            prof -= np.mean(prof[:50])
            profilesBSRH.append(prof)

    # BWS profiles
    for k in range(len(bunchN)):
        bws_index = 0
        while bunchN[k] != blist[bws_index] and bws_index < maxBunch:
            bws_index += 1
        if bws_index >= maxBunch:
            profilesBWS.append([])
        else:
            y = Proj_in[k * count:(k + 1) * count]
            y -= np.mean(y[:50])
            profilesBWS.append(moving_average(np.asarray(y), 3))

    # Align and fit
    Bws, Bsrh = [], []
    for idx in range(len(profilesBWS)):
        x = BWS_x
        y = profilesBWS[idx]
        out, _ = curve_fit(gaussian, x, y, (1000, 0.5, 0.35))
        x = (x - out[1]) * ratio
        newY = np.interp(newX, x, y)
        Bws.append(newY)

    for idx in range(len(profilesBSRH)):
        y = np.asarray(profilesBSRH[idx])
        x = np.arange(len(y))
        out, _ = curve_fit(gaussian, x, y, (60000, 480, 25.0))
        x -= out[1]
        Bsrh.append([x, y])

    return Bws, Bsrh, BWS_x, newX

def optimize_and_plot(Bws, Bsrh, crop, ratio, direction):
    newX = np.linspace(-4, 4, 1000)
    initGuess = [0.4, 42.5, 0.07, 1.4]
    result = minimize(CoolFunc, initGuess, args=(Bws, Bsrh, crop),
                      bounds=((0.1, 0.8), (10, 80), (0.005, 2), (0.1, 4)))
    psf, pix = result.x[0], result.x[1]
    print(f'{direction} PSF={psf}, Pixel size={pix}')

    tmpPSF = gaussian(newX, 1 / (psf * np.sqrt(2 * np.pi)), 0, psf)
    tmpPSF /= np.sum(tmpPSF)

    fig, ax = plt.subplots(len(Bsrh), figsize=(5, 10))
    fig.suptitle(f"{direction} Comparison of BSRH and BWS Profiles")

    for i in range(len(Bsrh)):
        bsrh = np.interp(newX, pix / 1000 * Bsrh[i][0], Bsrh[i][1])
        bws = np.convolve(Bws[i], tmpPSF, 'same')
        scale = (np.sum(bws) / np.sum(bsrh)) * bsrh
        ax[i].plot(scale, label='BSRH')
        ax[i].plot(bws, label='BWS')
        ax[i].legend()
        ax[i].set_title(f"Bunch: {i}")

    plt.show()
    return psf, pix

#--------------------- MAIN ----------------------#

DATA_PATH_BWS = '/user/jpucek/BWS/Data'
DATA_PATH_BSRH = '/user/jpucek/BSRH/Data'

dctx = zstd.ZstdDecompressor()
japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')

hRatio = 1.008
vRatio = 0.9363
bunchN = []

for direction in ['H', 'V']:
    bws_path = os.path.join(DATA_PATH_BWS, 'Horizontal' if direction == 'H' else 'Vertical',
                            time.strftime('%Y/%m/%d/', time.localtime()))
    BWS = load_BWS_file(bws_path, dctx)
    if not BWS: continue

    timeBWS = dt.datetime.fromtimestamp(BWS.value['gmtTimestampAcquireIn_ns'] / 1e9)
    BSRH = find_matching_BSRH(os.path.join(DATA_PATH_BSRH, time.strftime('%Y/%m/%d/', time.localtime())), timeBWS, dctx)
    if not BSRH: continue

    ratio = hRatio if direction == 'H' else vRatio
    Bws, Bsrh, BWS_x, newX = process_profiles(BSRH, BWS, direction, bunchN, ratio)
    crop = 570 if direction == 'H' else -1
    psf, pix = optimize_and_plot(Bws, Bsrh, crop, ratio, 'Horizontal' if direction == 'H' else 'Vertical')
    print(f"{direction} Results: PSF={psf}, Pixel Size={pix}")
