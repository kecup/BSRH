import dill as pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import pyjapc
import zstandard as zstd
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import pandas as pd

# ---------------- CONFIG ----------------
directory = '/user/jpucek/BSRH/Data/2025/09/26/'
files = sorted(os.listdir(directory))
dctx = zstd.ZstdDecompressor()
japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')

bunch = 55
# file_range = range(110, 112)
file_range = range(2)

maskX, maskY, maskR = 110, 110, 50  # Excluded signal region wrt. x1, x2,
x1, x2 = 390, 690#284, 734
y1, y2 = 100, 400#124, 424
# -----------------------------------------

# Frame map: global_frame_index -> (file_idx, local_idx)
frame_map, file_idx_map = [], {}
for local_file_idx, file_idx in enumerate(file_range):
    file_path = os.path.join(directory, files[file_idx])
    file_idx_map[file_idx] = file_path
    with open(file_path, "rb") as f:
        with dctx.stream_reader(f) as reader:
            try:
                a = pickle.load(reader)
                local_valid_idx = 0
                for frame in a:
                    gate = frame.device_data["GATE-INT"]
                    # if gate["delaySlots"] == bunch and gate["pulseLength"] >= 400:
                    frame_map.append((file_idx, local_valid_idx))
                    local_valid_idx += 1
            except Exception as e:
                print(f"Error reading {files[file_idx]}: {e}")

print(f"Total valid frames: {len(frame_map)}")

# ---- Utility: projection & smoothing ----
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

def gconst(x, amp, cen, wid, const):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2)) + const

def process_projection(proj):
    proj = gaussian_filter1d(proj, sigma=2)
    proj[proj < 0] = 0
    # Keep only longest positive chain
    positive = proj > 0
    best_len, best_range, start = 0, (0, len(proj)), None
    for i, val in enumerate(positive):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start > best_len:
                best_len = i - start
                best_range = (start, i)
            start = None
    if start is not None and len(proj) - start > best_len:
        best_range = (start, len(proj))
    proj[:best_range[0]] = 0
    proj[best_range[1]:] = 0
    return proj

# ---- Chi2 displacement search (skip mask region) ----
def chi2_displacement(proj1, proj2, max_shift=15, axis="x"):
    x = np.arange(len(proj1))
    f2 = interp1d(x, proj2, bounds_error=False, fill_value=0.0)

    # exclude masked central region
    if axis == "x":
        exclude = np.arange(maskX - maskR, maskX + maskR)
    else:
        exclude = np.arange(maskY - maskR, maskY + maskR)

    valid = np.ones_like(proj1, dtype=bool)
    valid[exclude] = False

    chi2_vals = []

    for shift in range(-max_shift, max_shift + 1): # We should make sure that we use the same amount of points for each chi2, but ok for now...
        shifted = f2(x - shift)
        diff = (proj1 - shifted)[valid]
        chi2 = np.sum(diff**2)
        chi2_vals.append((shift, chi2))

    best_shift = min(chi2_vals, key=lambda t: t[1])[0]
    return best_shift

# ---- Runtime cache ----
cached_file_idx, cached_images = None, []
shifts_x, shifts_y = [], []

def load_and_cache_file(file_idx):
    global cached_file_idx, cached_images
    cached_images = []
    file_path = file_idx_map[file_idx]
    with open(file_path, "rb") as f:
        with dctx.stream_reader(f) as reader:
            a = pickle.load(reader)
            for frame in a:
                gate = frame.device_data["GATE-INT"]
                #if gate["delaySlots"] == bunch and gate["pulseLength"] >= 400:
                image = frame.cam_data["image2D"]
                #image = image - np.mean(image[:100, :100])

                proj_x = np.sum(image[y1:y2, :], axis=0)
                proj_y = np.sum(image[:, x1:x2], axis=1)
                proj_x = process_projection(proj_x)
                proj_y = process_projection(proj_y)

                cached_images.append((image, proj_x, proj_y))
    cached_file_idx = file_idx

print(len(cached_images))

def gaussian_fit_center(projection, mask_pos, mask_rad, axis_len):
    x = np.arange(axis_len)

    # Use only "tails" outside masked region
    valid_idx = np.r_[0:mask_pos-mask_rad, mask_pos+mask_rad:axis_len]
    y = projection[valid_idx]

    try:
        popt, _ = curve_fit(
            gaussian, valid_idx, y,
            p0=[np.max(y), mask_pos, mask_rad]
        )
        return popt[1]  # fitted center
    except:
        return np.nan

# ---- Animation ----
fig, (ax_img, ax_px, ax_py) = plt.subplots(1, 3, figsize=(15, 5))
tst = ax_img.imshow(np.zeros((y2-y1, x2-x1)), cmap="jet", vmin=800, vmax=2000) #, norm="log")
plt.colorbar(tst, ax=ax_img)

line_px, = ax_px.plot([], [], label="Projection X")
line_py, = ax_py.plot([], [], label="Projection Y")

#exclude_x1 = ax_px.axvline(maskX - maskR - x1, color="red", linestyle="--")
#exclude_x2 = ax_px.axvline(maskX + maskR - x1, color="red", linestyle="--")

#exclude_y1 = ax_py.axvline(maskY - maskR - y1, color="red", linestyle="--")
#exclude_y2 = ax_py.axvline(maskY + maskR - y1, color="red", linestyle="--")
ax_px.legend()
ax_py.legend()

shift_x, shift_y = 0, 0
shifts_data = []

def update(frame_number):
    global cached_file_idx, shift_x, shift_y
    file_idx, local_idx = frame_map[frame_number]

    if cached_file_idx != file_idx:
        load_and_cache_file(file_idx)

    image, proj_x, proj_y = cached_images[local_idx]

    if frame_number > 0:
        _, prev_px, prev_py = cached_images[max(0, local_idx - 1)]
        dx = chi2_displacement(prev_px, proj_x, axis="x")
        dy = chi2_displacement(prev_py, proj_y, axis="y")
        shift_x += dx
        shift_y += dy

    # Apply cumulative shift
    shifted = np.roll(image, shift=(shift_y, shift_x), axis=(0, 1))

    # --- CROP the shifted image here ---
    cropped = shifted[y1:y2, x1:x2]

    # Update the image display with the cropped version
    tst.set_data(cropped)

    # Recalculate projections on cropped region only
    proj_x_shifted = np.sum(cropped, axis=0)
    proj_y_shifted = np.sum(cropped, axis=1)
    proj_x_shifted = process_projection(proj_x_shifted)
    proj_y_shifted = process_projection(proj_y_shifted)

    # Update plots
    line_px.set_data(np.arange(len(proj_x_shifted)), proj_x_shifted)
    line_py.set_data(np.arange(len(proj_y_shifted)), proj_y_shifted)

    shifts_x.append(shift_x)
    shifts_y.append(shift_y)

    ax_px.relim(); ax_px.autoscale_view()
    ax_py.relim(); ax_py.autoscale_view()

    # --- Gaussian fit centers ---
    proj_x = np.sum(cropped, axis=0)[100:200]
    proj_y = np.sum(cropped, axis=1)[100:200]

    popt, _ = curve_fit(gconst, range(len(proj_x)), proj_x, p0=[-200000, 50, 50, 400000])
    print('popt = ', popt)
    cx_gauss = popt[1] #np.argmin(proj_x[100:200])#gaussian_fit_center(proj_x, maskX, maskR, proj_x.shape[0])

    popt, _ = curve_fit(gconst, range(len(proj_y)), proj_y, p0=[-200000, 50, 50, 400000])
    cy_gauss = popt[1] #np.argmin(proj_y[100:200])#gaussian_fit_center(proj_y, maskY, maskR, proj_y.shape[0])

    # Save shifts for comparison
    shifts_data.append([frame_number, shift_x, shift_y, cx_gauss, cy_gauss])

    return (tst, line_px, line_py)

# ani = animation.FuncAnimation(fig=fig, func=update, frames=len(frame_map), interval=100, repeat=False)
ani = animation.FuncAnimation(fig=fig, func=update, frames=160, interval=100, repeat=False)

writer = animation.PillowWriter(fps=20, metadata=dict(artist="Me"))
ani.save(f"Video_occulterOnly_{bunch}.gif", writer=writer)

df = pd.DataFrame(shifts_data, columns=["frame", "chi2_dx", "chi2_dy", "gauss_cx", "gauss_cy"])
plt.figure()
# plt.plot(df["frame"], df["chi2_dx"], label="Chi2 dx", color="blue")
plt.plot(df["frame"], df["gauss_cx"]-df['gauss_cx'][0], label="Gaussian cx", color="orange", linestyle="--")

plt.figure()
# plt.plot(df["frame"], df["chi2_dy"], label="Chi2 dy", color="green")
plt.plot(df["frame"], df["gauss_cy"]-df['gauss_cy'][0], label="Gaussian cy", color="red", linestyle="--")

plt.show()