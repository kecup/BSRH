import dill as pickle
import pyjapc
import zstandard as zstd
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.restoration import richardson_lucy
from scipy.optimize import curve_fit

def gaussian(x, amp, cen, wid):  # , const):
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))  # +const



def get_background(directory: str, file, event):
    f = open(directory + file, 'rb')
    with dctx.stream_reader(f) as reader:
        file_data = pickle.load(reader)
    image = file_data[event].cam_data['image2D']
    num_of_pulses = file_data[event].device_data['GATE-INT']['numberOfPulses']
    return image, num_of_pulses


def get_run_data(directory: str, files, run: str):
    data = []
    for file in files:
        if run in file:
            with open(directory + file, 'rb') as f:
                with dctx.stream_reader(f) as reader:
                    file_data = pickle.load(reader)
                    data.append(file_data)
    return data


def graph_cpc(data):
    n_runs = len(data)
    n_cols = 2
    n_rows = (n_runs + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    all_delay_slots = sorted({int(mask.split("_")[1]) for run in data.values() for mask in run if mask.startswith("mask_")})
    colors = plt.cm.get_cmap("tab10", len(all_delay_slots))  # dynamic colormap

    for idx, (run_num, run_data) in enumerate(data.items()):
        ax = axes[idx]
        events = run_data['events']
        cpc = run_data['cpcs']

        for mask_idx, delay in enumerate(all_delay_slots):
            mask_name = f"mask_{delay}"
            mask = run_data.get(mask_name, [])
            ax.scatter(
                [events[i] for i in mask],
                [cpc[i] for i in mask],
                color=colors(mask_idx),
                label=f"bunch {delay}",
                s=12
            )

        ax.set_title(f'{run_num}')
        ax.set_xlabel('Events')
        ax.set_ylabel('CPC')
        ax.legend()

    for i in range(n_runs, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def compute_bg(directory: str, files, runs: list[str], sigma_x_um, sigma_y_um, pixel_size_x, pixel_size_y, size_x, size_y, sigma):
    results = {}
    mask = []

    for run in runs:
        print("Analysing " + run)
        run_data = get_run_data(directory, files, run)
        mask_dict = {}  # Key: delay_slots, Value: list of event indices
        event_global = 0
        for file_data in run_data:
            for event in file_data:
                img = event.cam_data['image2D']
                time = event.timestamp
                delay_slots = int(event.device_data["GATE-INT"]["delaySlots"])
                pulse_length = int(event.device_data["GATE-INT"]["pulseLength"])
                cm_x = event.device_data['CMASK-X']
                #cm_y = event.device_data['CMASK-Y']
                img_charge = np.sum(event.device_data["FBCT1"]["averageBunchIntensities"][delay_slots:delay_slots + pulse_length // 4][0])

                fArr = event.device_data['FILTERS.B2']['channelState']
                in_filter = pow(12.6, 3 * fArr[0] + 2 * fArr[1] + fArr[2])
                magic_const = 5651
                num_of_pulses = int(event.device_data["GATE-INT"]["numberOfPulses"])

                # Subtract background if needed:
                img = np.asarray(img) - np.mean(img[:50,:50])
                #img = img - (num_of_pulses / bg_pulses) * bg_image

                img_sum = np.sum(img[154:454, 264:644])
                #img_sum = np.sum(img)
                cpc = (img_charge * num_of_pulses) / (img_sum * magic_const * in_filter)

                if delay_slots not in mask_dict:
                    mask_dict[delay_slots] = {'time': [], 'bg_x': [], 'bg_y': [], 'charge_x': [], 'charge_y': []}

                if cm_x > 61.0:

                    time, bg_x, bg_y, charge_x, charge_y = process_image(img, img_charge, time, sigma_x_um, sigma_y_um, pixel_size_x, pixel_size_y, size_x, size_y, sigma)

                    mask_dict[delay_slots]['time'].append(time)
                    mask_dict[delay_slots]['bg_x'].append(bg_x)
                    mask_dict[delay_slots]['bg_y'].append(bg_y)
                    mask_dict[delay_slots]['charge_x'].append(charge_x)
                    mask_dict[delay_slots]['charge_y'].append(charge_y)

                print("Done event: ", event_global)
                event_global += 1

        results[run] = mask_dict

    return results


def plot_example_images(first_images):
    for run, delays in first_images.items():
        n = len(delays)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, (delay, (img, delay_val)) in zip(axes, sorted(delays.items())):
            ax.imshow(img, cmap='jet')
            ax.set_title(f"Run: {run}\nDelay: {delay_val}", fontsize=10)
            ax.text(10, 20, f"Delay: {delay_val}", color='red', fontsize=9, weight='bold', backgroundcolor='white')
            ax.axis('off')
        fig.suptitle(f"Example Images from {run}")
        plt.tight_layout()
        plt.show()

def gaussian_psf(shape, sigma_x_um, sigma_y_um, pixel_size_x_um, pixel_size_y_um):
    """Asymmetric 2D Gaussian PSF with per-axis pixel size."""
    h, w = shape
    x = (np.arange(w) - w // 2) * pixel_size_x_um
    y = (np.arange(h) - h // 2) * pixel_size_y_um
    X, Y = np.meshgrid(x, y)
    psf = np.exp(-0.5 * ((X / sigma_x_um)**2 + (Y / sigma_y_um)**2))
    return psf / np.sum(psf)

def deconvolve_image(image, sigma_x_um, sigma_y_um, pixel_size_x_um, pixel_size_y_um, iterations=5):
    psf = gaussian_psf(image.shape, sigma_x_um, sigma_y_um, pixel_size_x_um, pixel_size_y_um)
    image = np.clip(image.astype(float), 0, None)
    return richardson_lucy(image, psf, num_iter=iterations, clip = False)


def fit_gaussian_1d_projection(projection):
    x = np.arange(len(projection))

    try:
        popt, _ = curve_fit(gaussian, x, projection, (projection.max(), x[np.argmax(projection)], len(projection)/4))
        _, mu, _ = popt
    except RuntimeError:
        mu = x[np.argmax(projection)]  # fallback: peak index
    return mu

def find_beam_center(image):
    """Find beam center using image moments."""
    proj_x = np.sum(image, axis=0)  # sum over rows (vertical), profile along x
    proj_x = proj_x - np.mean(proj_x[:50])
    proj_y = np.sum(image, axis=1)  # sum over columns (horizontal), profile along y
    proj_y = proj_y - np.mean(proj_y[:50])

    cx = fit_gaussian_1d_projection(proj_x)
    cy = fit_gaussian_1d_projection(proj_y)
    return int(cx), int(cy)

def integrate_projection(image, cx, px, cy, py, dlx = -1, dhx = -1, dly = -1, dhy = -1): ## This has to be adapted...
    """Project charge from center to edge in X and Y."""
    horizontal_proj = np.sum(image[:, :], axis=0)
    vertical_proj = np.sum(image[:, :], axis=1)

    horizontal_proj = horizontal_proj - np.mean(horizontal_proj[:50])
    vertical_proj = vertical_proj - np.mean(vertical_proj[:50])

    #Find limits:
    if dly > 0:
        ll = int(cy-py-dly)
    else:
        ll = int(cy-py)
        while vertical_proj[ll] > 0 and ll > 0:
            ll-= 1
        ll += 1

    if dhy > 0:
        rl = int(cy+py+dhy)
    else:
        rl = int(cy+py)
        while vertical_proj[rl] > 0 and rl < len(vertical_proj)-1:
            rl+=1
        rl -= 1

    charge_y = np.sum(vertical_proj[ll:int(cy-py)])+np.sum(vertical_proj[int(cy+py):rl])
    total_y = np.sum(vertical_proj[ll:rl])

    dly = cy-py-ll
    dhy = rl - (cy + py)

    # Find limits:
    if dlx > 0:
        ll = int(cx-px-dlx)
    else:
        ll = int(cx - px)
        while horizontal_proj[ll] > 0 and ll > 0:
            ll -= 1
        ll+=1

    if dhx > 0:
        rl = int(cx+px+dhx)
    else:
        rl = int(cx + px)
        while horizontal_proj[rl] > 0 and rl < len(horizontal_proj)-1:
            rl += 1
        rl -= 1
    charge_x = np.sum(horizontal_proj[ll:int(cx - px)]) + np.sum(horizontal_proj[int(cx + px):rl])
    total_x = np.sum(horizontal_proj[ll:rl])


    if total_x == 0:
        print("Here")
        charge_x = 0
    else:
        charge_x = charge_x/total_x

    if total_y == 0:
        print("There")
        plt.imshow(image)
        charge_y = 0
    else:
        charge_y = charge_y/total_y

    dlx = cx-px-ll
    dhx = rl - (cx + px)

    return charge_x, charge_y, vertical_proj, horizontal_proj, total_x, total_y, dly, dhy, dlx, dhx

def process_image(first_image, fbct, time, sigma_x_um, sigma_y_um, pixel_size_x, pixel_size_y, size_x, size_y, sigma):
    result = {}
    px = sigma * size_x / pixel_size_x
    py = sigma * size_y / pixel_size_y

    cx1, cy1 = find_beam_center(first_image)
    charge_x_light, charge_y_light, proj_y1, proj_x1, total1_x, total1_y, dly, dhy, dlx, dhx = integrate_projection(first_image, cx1, px, cy1, py)
    deconv = deconvolve_image(first_image, sigma_x_um, sigma_y_um, pixel_size_x, pixel_size_y, 10)
    cx, cy = find_beam_center(deconv)
    charge_x, charge_y, proj_y2, proj_x2, total_x, total_y, _, _, _, _ = integrate_projection(deconv, cx, px, cy, py, dlx, dhx, dly, dhy)

    bg_x = (charge_x_light - charge_x) * fbct
    bg_y = (charge_y_light - charge_y) * fbct

        #     run_result[delay] = {
        #         "deconvolved": deconv,
        #         "time": time,
        #         "center": (cx, cy),
        #         "distance_x": (dlx, dhx),
        #         "distance_y": (dly, dhy),
        #         "charge_x_light": charge_x_light,
        #         "charge_y_light": charge_y_light,
        #         "charge_x_proton": charge_x,
        #         "charge_y_proton": charge_y,
        #         "proj_x_light": proj_x1,
        #         "proj_x_proton": proj_x2,
        #         "proj_y_light": proj_y1,
        #         "proj_y_proton": proj_y2,
        #         "total_x_light": total1_x,
        #         "total_y_light": total1_y,
        #         "total_x_proton": total_x,
        #         "total_y_proton": total_y,
        #         "fbct_charge": fbct_charge
        #         #"limits_x_light": (ll1, rl1),
        #         #"limits_x_proton": (ll, rl)
        #     }
        #
        # result[run] = run_result
    return time, bg_x, bg_y, charge_x*fbct, charge_y*fbct

if __name__ == "__main__":
    runs = ['run_02', 'run_04', 'run_09', 'run_11', 'run_13']
    japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')

    dctx = zstd.ZstdDecompressor()
    directory = '/user/jpucek/BSRH/Data/2025/06/20/'
    files = sorted(os.listdir(directory))

     # so first_images contains the last images without occulter, so we can then calculate the BG by performing a 2D deconvolution...
    #graph_cpc(results)

    size_x = 263.0 # sizes of the beam at the dipole location
    size_y = 361.0
    sigma_x_um = 450 ## sigma of the PSF in micrometers
    sigma_y_um = 250
    px_x = 18.5 # pixel sizes of the images, might have to change for different PSF...
    px_y = 17.5

    results = compute_bg(directory, files, runs, sigma_x_um, sigma_y_um, px_x, px_y, size_x, size_y, 3.5)

    f = open("BG_results_v8.pkl", 'wb')
    pickle.dump(results, f)
    f.close()