import sys
import os
import argparse
import datetime
import numpy as np
import dill as pickle
import zstandard as zstd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Initialize JAPC globally to allow unpickling of custom pyjapc objects
try:
    import pyjapc
    _global_japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
except ImportError:
    _global_japc = None

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Halo Trends from offline BSRH files")
    parser.add_argument("--directory", type=str, default="/user/jpucek/BSRH/Data/2025/08/04/", help="Directory containing .pkl.zst files")
    parser.add_argument("--bunches", type=int, nargs="*", default=None, help="Specific bunch slots to plot. If omitted, plots all discovered bunches.")
    parser.add_argument("--out", type=str, default="halo_trends.png", help="Output plot filename")
    return parser.parse_args()

def remove_outliers(data, window_size=10, threshold=3.0):
    """
    Removes outliers using a rolling variance/std-dev filter.
    window_size=10 looks 10 elements forward and backward (total window ~20).
    Values outside threshold * local_std are replaced with linear interpolation.
    """
    data = np.array(data, dtype=float)
    if len(data) == 0:
        return data

    cleaned = data.copy()

    # 1. Identify outliers
    for i in range(len(data)):
        start = max(0, i - window_size)
        end = min(len(data), i + window_size + 1)
        window = np.concatenate((data[start:i], data[i+1:end]))

        if len(window) < 2:
            continue

        local_mean = np.mean(window)
        local_std = np.std(window)

        if abs(data[i] - local_mean) > threshold * local_std:
            cleaned[i] = np.nan

    # 2. Linearly interpolate NaNs
    nans = np.isnan(cleaned)
    if np.any(nans):
        x = lambda z: z.nonzero()[0]
        cleaned[nans] = np.interp(x(nans), x(~nans), cleaned[~nans])

    return cleaned

def analyze_event(image, fbct_charge, num_pulses, background_image):
    # Hardcoded parameters based on OnlineGUI
    maskX = 474
    maskY = 307
    maskR = 55
    mask_factor = 8.3
    sigma_val = 3.5
    intensifier_gain = 5651.0

    x1, x2 = 234, 684
    y1, y2 = 154, 454

    try:
        bg_subtracted = image - background_image
    except ValueError:
        return None, None

    y_indices, x_indices = np.ogrid[:bg_subtracted.shape[0], :bg_subtracted.shape[1]]
    mask_area = (x_indices - maskX)**2 + (y_indices - maskY)**2 <= maskR**2
    bg_subtracted[mask_area] = bg_subtracted[mask_area] * mask_factor

    img_sum = np.sum(bg_subtracted)
    if img_sum > 0 and num_pulses > 0:
        cPc = (fbct_charge * num_pulses) / (img_sum * intensifier_gain)
    else:
        return None, None

    try:
        projection_x = np.sum(bg_subtracted[y1:y2, :], axis=0)
        projection_y = np.sum(bg_subtracted[:, x1:x2], axis=1)
    except Exception:
        return None, None

    x_temp = []
    x_temp.extend(range(int(max(0, maskX - maskR))))
    x_temp.extend(range(int(min(len(projection_x), maskX + maskR)), len(projection_x)))

    resX_center = maskX
    if len(x_temp) > 0:
        try:
            resX, _ = curve_fit(gaussian, x_temp, projection_x[x_temp], (45000, maskX, 35.0))
            if not (maskX - maskR < resX[1] < maskX + maskR):
                resX_center = resX[1]
        except:
            pass

    y_temp = []
    y_temp.extend(range(int(max(0, maskY - maskR))))
    y_temp.extend(range(int(min(len(projection_y), maskY + maskR)), len(projection_y)))

    resY_center = maskY
    if len(y_temp) > 0:
        try:
            resY, _ = curve_fit(gaussian, y_temp, projection_y[y_temp], (45000, maskY, 15.0))
            if not (maskY - maskR < resY[1] < maskY + maskR):
                resY_center = resY[1]
        except:
            pass

    pixelSizeH = 19.6
    pixelSizeV = 20.3
    SIGMA_TO_MM_X = 263.0
    SIGMA_TO_MM_Y = 361.0

    xl1 = int(resX_center - sigma_val * SIGMA_TO_MM_X / pixelSizeH)
    xr1 = int(resX_center + sigma_val * SIGMA_TO_MM_X / pixelSizeH)

    chargeX = 0
    if xl1 > 0:
        chargeX += np.sum(projection_x[:xl1])
    if xr1 < len(projection_x):
        chargeX += np.sum(projection_x[xr1:])

    haloX = chargeX * cPc * intensifier_gain / num_pulses

    yl1 = int(resY_center - sigma_val * SIGMA_TO_MM_Y / pixelSizeV)
    yr1 = int(resY_center + sigma_val * SIGMA_TO_MM_Y / pixelSizeV)

    chargeY = 0
    if yl1 > 0:
        chargeY += np.sum(projection_y[:yl1])
    if yr1 < len(projection_y):
        chargeY += np.sum(projection_y[yr1:])

    haloY = chargeY * cPc * intensifier_gain / num_pulses

    return haloX, haloY

def main():
    args = parse_args()
    directory = args.directory
    bunches_to_plot = args.bunches

    if not os.path.isdir(directory):
        print(f"Error: Directory {directory} not found.")
        sys.exit(1)

    files = sorted([f for f in os.listdir(directory) if f.endswith('.pkl.zst')])
    if not files:
        print(f"No .pkl.zst files found in {directory}")
        sys.exit(1)

    print(f"Found {len(files)} files to process.")

    dctx = zstd.ZstdDecompressor()

    trends_h = {}
    trends_v = {}

    background_image = None

    for filename in files:
        full_path = os.path.join(directory, filename)
        print(f"Processing {filename}...")

        try:
            with open(full_path, 'rb') as f:
                with dctx.stream_reader(f) as reader:
                    data_chunk = pickle.load(reader)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            continue

        for event in data_chunk:
            try:
                image2D = event.cam_data['image2D']
                device_data = event.device_data

                delay_slots = device_data['GATE-INT']['delaySlots']
                num_pulses = device_data['GATE-INT']['numberOfPulses']
                bunch_index = int(delay_slots)

                fbct_list = device_data['FBCT1']['averageBunchIntensities']
                if 0 <= bunch_index < len(fbct_list):
                    fbct_charge = fbct_list[bunch_index]
                else:
                    fbct_charge = 0.0

                timestamp = event.timestamp # datetime object or POSIX timestamp
                if isinstance(timestamp, (int, float)):
                    dt_obj = datetime.datetime.fromtimestamp(timestamp)
                else:
                    dt_obj = timestamp # Assume it's already a datetime object

            except Exception as e:
                # Malformed event
                continue

            # Filter by specific bunches if requested
            if bunches_to_plot is not None and bunch_index not in bunches_to_plot:
                continue

            if fbct_charge <= 0:
                background_image = image2D.copy()
                continue

            if background_image is None:
                continue

            haloX, haloY = analyze_event(image2D, fbct_charge, num_pulses, background_image)

            if haloX is not None and haloY is not None:
                if bunch_index not in trends_h:
                    trends_h[bunch_index] = []
                    trends_v[bunch_index] = []

                trends_h[bunch_index].append((dt_obj, haloX))
                trends_v[bunch_index].append((dt_obj, haloY))

    # Plotting
    if not trends_h:
        print("No valid data points found to plot.")
        sys.exit(0)

    print(f"Plotting data for {len(trends_h)} bunches...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for bunch_idx in sorted(trends_h.keys()):
        # H Data
        data_h = trends_h[bunch_idx]
        data_h.sort(key=lambda x: x[0]) # Sort by time
        times_h = [d[0] for d in data_h]
        vals_h = [d[1] for d in data_h]
        vals_h_filtered = remove_outliers(vals_h, window_size=10, threshold=3.0)
        ax1.plot(times_h, vals_h_filtered, marker='o', linestyle='-', markersize=2, label=f"Bunch {bunch_idx}")

        # V Data
        data_v = trends_v[bunch_idx]
        data_v.sort(key=lambda x: x[0])
        times_v = [d[0] for d in data_v]
        vals_v = [d[1] for d in data_v]
        vals_v_filtered = remove_outliers(vals_v, window_size=10, threshold=3.0)
        ax2.plot(times_v, vals_v_filtered, marker='o', linestyle='-', markersize=2, label=f"Bunch {bunch_idx}")

    ax1.set_title("Horizontal Halo Trends")
    ax1.set_ylabel("Halo Charge")
    ax1.grid(True)
    if len(trends_h) <= 20: # Only show legend if not too many bunches
        ax1.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    ax2.set_title("Vertical Halo Trends")
    ax2.set_ylabel("Halo Charge")
    ax2.set_xlabel("Time")
    ax2.grid(True)
    if len(trends_v) <= 20:
        ax2.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"Plots saved to {args.out}")

if __name__ == "__main__":
    main()
