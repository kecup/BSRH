import dill as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

from matplotlib.widgets import CheckButtons

class CurveVisibilityToggler:
    """
    Add interactive checkboxes to toggle line visibility by label,
    without layout shifting, on any given figure.
    """

    def __init__(self, fig, linewidth=1.5, box_position=None, font_size=10):
        self.fig = fig
        self.linewidth = linewidth
        self.label_map = {}  # Maps display label -> list of lines (line objects unique to this fig)

        label_counts = {}

        # Collect all labeled lines from all axes
        for ax in fig.axes:
            for line in ax.get_lines():
                label = line.get_label()
                if label.startswith("_"):  # ignore default/anonymous labels
                    continue
                # Disambiguate repeated labels in the same figure
                count = label_counts.get(label, 0)
                unique_label = label if count == 0 else f"{label} ({count})"
                label_counts[label] = count + 1
                self.label_map.setdefault(unique_label, []).append(line)

        # Prepare check buttons
        labels = list(self.label_map.keys())
        visibility = [True] * len(labels)

        if box_position is None:
            height = min(0.05 * len(labels), 0.8)
            box_position = (0.8, 0.9 - height, 0.18, height)

        rax = self.fig.add_axes(box_position)
        self.check = CheckButtons(rax, labels, visibility)

        # Force widget to respond by drawing and connecting event loop
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Update label font size
        for text in self.check.labels:
            text.set_fontsize(font_size)

        self.check.on_clicked(self.toggle_visibility)
        print(f"Adding checkboxes to figure {self.fig.number} with labels: {labels}")

    def toggle_visibility(self, label):
        for line in self.label_map[label]:
            current_lw = line.get_linewidth()
            if current_lw == 0:
                line.set_linewidth(self.linewidth)
            else:
                line.set_linewidth(0)
        self.fig.canvas.draw_idle()


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

InFile = open('HaloCharge_v4.pkl', 'rb')
results = pickle.load(InFile)
InFile.close()

g = open("AvgBG5.pkl", 'rb') #From this file we want to interpolate the bg...
BG = pickle.load(g)
g.close()

h = open('AvgHalo5.pkl','rb')
CH = pickle.load(h)
h.close()

bws = open('BWSCharge_v3.pkl', 'rb')
BWS = pickle.load(bws)



keys = list(results.keys())

key_dict = {20:0, 500:1, 1050:2, 1250:3, 1800:4, 2500:5, 2874:6}

for index, key in enumerate(keys):

    bws_x = BWS[0][key_dict[key]]
    bws_y = BWS[1][key_dict[key]]
    bws_q = BWS[2][key_dict[key]]

    y = results[key]['haloH']
    x = results[key]['timeH']

    x_sorted, y_sorted = zip(*sorted(zip(x, y)))
    y_clean = remove_outliers(y_sorted, 50, 2.5)
    y_sorted = list(y_sorted)
    tmpStamp = [dt.timestamp() for dt in BG[key][0]]

    y_corrected = np.asarray(y_sorted) - (y_sorted[0]-bws_y[0]) # Setting the BWS equal to the first BSRH measurement
    #y_corrected = y_sorted#[y - np.interp(x.timestamp(), tmpStamp, BG[key][1]) for x, y in zip(x_sorted, y_sorted) ]

    y_clean = y_corrected #remove_outliers(y_corrected, 50, 2.5)

    plt.figure(1)
    plt.plot(x_sorted,y_clean, label = str(key))
    plt.plot(bws_x, bws_y, label = 'BWS b.'+str(key))
    plt.plot(bws_x, bws_q, label='q-Gauss b.' + str(key))



    y = results[key]['haloV']
    x = results[key]['timeV']

    bws_x = BWS[3][key_dict[key]]
    bws_y = BWS[4][key_dict[key]]
    bws_q = BWS[5][key_dict[key]]

    x_sorted, y_sorted = zip(*sorted(zip(x, y)))
    y_sorted = list(y_sorted)
    tmpStamp = [dt.timestamp() for dt in BG[key][0]]

    y_corrected = y_sorted#[ y - np.interp(x.timestamp(), tmpStamp, BG[key][3]) for x, y in zip(x_sorted, y_sorted) ]

    y_clean = remove_outliers(y_corrected, 50, 2.5)

    plt.figure(2)
    plt.plot(x_sorted, y_clean, label = str(key))
    plt.plot(bws_x, bws_y, label='BWS b.' + str(key))
    plt.plot(bws_x, bws_q, label='q-Gauss b.' + str(key))

#plt.ion()

plt.figure(1)
plt.legend()
plt.title("Horizontal halo charge, PSF = 450 um")
CurveVisibilityToggler(plt.figure(1))

plt.figure(2)
plt.legend()
plt.title("Vertical halo charge, PSF = 250 um")
CurveVisibilityToggler(plt.figure(2))

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