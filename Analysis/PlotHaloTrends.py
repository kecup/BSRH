import sys
import argparse
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt

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

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Halo Trends from extracted data")
    parser.add_argument("--data", type=str, default="halo_trends_data.pkl", help="Input extracted data pickle file")
    parser.add_argument("--out", type=str, default="halo_trends.png", help="Output plot filename")
    return parser.parse_args()

def main():
    args = parse_args()

    try:
        with open(args.data, 'rb') as f:
            data_dict = pickle.load(f)
    except Exception as e:
        print(f"Failed to load data from {args.data}: {e}")
        sys.exit(1)

    trends_h = data_dict.get('trends_h', {})
    trends_v = data_dict.get('trends_v', {})

    if not trends_h:
        print("No valid data points found in the dataset.")
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
    if len(trends_h) <= 20:
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
