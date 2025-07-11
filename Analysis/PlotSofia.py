import dill as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

InFile = open('HaloCharge.pkl', 'rb')
results = pickle.load(InFile)
InFile.close()

#g = open("AvgBG.pkl", 'rb') #From this file we want to interpolate the bg...
#BG = pickle.load(g)
#g.close()

keys = list(results.keys())

for key in keys:
    y = results[key]['haloH']
    x = results[key]['timeH']

    x_sorted, y_sorted = zip(*sorted(zip(x, y)))
    y_clean = remove_outliers(y_sorted, 50, 2.5)

    plt.figure(1)
    plt.plot(x_sorted,y_clean, label = str(key))

    y = results[key]['haloV']
    x = results[key]['timeV']

    x_sorted, y_sorted = zip(*sorted(zip(x, y)))
    y_clean = remove_outliers(y_sorted, 50, 2.5)

    plt.figure(2)
    plt.plot(x_sorted, y_clean, label = str(key))

plt.figure(1)
plt.legend()
plt.figure(2)
plt.legend()
plt.show()