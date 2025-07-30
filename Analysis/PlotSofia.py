import dill as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def remove_outliers(data, window_size=5, threshold=3.0):
    data = np.array(data)
    cleaned = data.copy()

    for i in range(len(data)):
        # Define the moving window (exclude current point)
        start = max(0, i - window_size)
        end = min(len(data), i + window_size + 1)
        window = np.concatenate((data[start:i], data[i + 1:end]))

        if len(window) == 0:
            continue

        mean = np.mean(window)
        std = np.std(window)

        if abs(data[i] - mean) > threshold * std:
            cleaned[i] = np.nan  # Or interpolate later, or delete

    cleaned = pd.Series(cleaned).interpolate(method='linear', limit_direction='both').to_numpy()

    return cleaned

InFile = open('HaloCharge_v2.pkl', 'rb')
results = pickle.load(InFile)
InFile.close()

g = open("AvgBG3.pkl", 'rb') #From this file we want to interpolate the bg...
BG = pickle.load(g)
g.close()

h = open('AvgHalo3.pkl','rb')
CH = pickle.load(h)
h.close()

bws = open('BWSCharge_v1.pkl', 'rb')
BWS = pickle.load(bws)



keys = list(results.keys())

for index, key in enumerate(keys):

    bws_x = BWS[0][index]
    bws_y = BWS[1][index]


    y = results[key]['haloH']
    x = results[key]['timeH']

    x_sorted, y_sorted = zip(*sorted(zip(x, y)))
    y_clean = remove_outliers(y_sorted, 50, 2.5)
    y_sorted = list(y_sorted)
    tmpStamp = [dt.timestamp() for dt in BG[key][0]]

    y_corrected = [
        y - np.interp(x.timestamp(), tmpStamp, BG[key][1])
        for x, y in zip(x_sorted, y_sorted)
    ]

    y_clean = remove_outliers(y_corrected, 50, 2.5)

    plt.figure(1)
    plt.plot(x_sorted,y_clean, label = str(key))
    plt.plot(bws_x, bws_y, label = 'BWS b.'+str(key))

    y = results[key]['haloV']
    x = results[key]['timeV']

    x_sorted, y_sorted = zip(*sorted(zip(x, y)))
    y_sorted = list(y_sorted)
    tmpStamp = [dt.timestamp() for dt in BG[key][0]]

    y_corrected = [
        y - np.interp(x.timestamp(), tmpStamp, BG[key][3])
        for x, y in zip(x_sorted, y_sorted)
    ]

    y_clean = remove_outliers(y_corrected, 50, 2.5)

    plt.figure(2)
    plt.plot(x_sorted, y_clean, label = str(key))

plt.figure(1)
plt.legend()
plt.title("Horizontal halo charge, PSF = 420 um")
plt.figure(2)
plt.legend()
plt.title("Vertical halo charge, PSF = 232 um")

keys = list(CH.keys())
for key in keys:
    x = CH[key][0]

    y = CH[key][1]
    y_err = CH[key][2]
    plt.figure(3)
    plt.errorbar(x, y, y_err, label = str(key))

    y = CH[key][3]
    y_err = CH[key][4]
    plt.figure(4)
    plt.errorbar(x, y, y_err, label = str(key))

plt.figure(3)
plt.legend()
plt.title("Horizontal deconvolved charge, PSF = 420 um")
plt.figure(4)
plt.legend()
plt.title("Vertical halo charge, PSF = 232 um")
plt.show()