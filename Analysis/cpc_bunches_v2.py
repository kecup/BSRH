import dill as pickle
import pyjapc
import zstandard as zstd
import numpy as np
import os
import matplotlib.pyplot as plt


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


def compute_cpc(directory: str, files, runs: list[str]):
    results = {}
    first_images = {}

    for run in runs:
        print("Analysing " + run)
        run_data = get_run_data(directory, files, run)
        mask_dict = {}  # Key: delay_slots, Value: list of event indices
        run_cpcs = []
        run_counter = 0
        bg_image, bg_pulses = get_background(directory, files[4], 2)

        for file_data in run_data:
            for event in file_data:
                img = event.cam_data['image2D']
                delay_slots = int(event.device_data["GATE-INT"]["delaySlots"])
                pulse_length = int(event.device_data["GATE-INT"]["pulseLength"])
                img_charge = np.sum(event.device_data["FBCT1"]["averageBunchIntensities"][delay_slots:delay_slots + pulse_length // 4][0])

                fArr = event.device_data['FILTERS.B2']['channelState']
                in_filter = pow(12.6, 3 * fArr[0] + 2 * fArr[1] + fArr[2])
                magic_const = 5651
                num_of_pulses = int(event.device_data["GATE-INT"]["numberOfPulses"])

                # Subtract background if needed:
                img = img - (num_of_pulses / bg_pulses) * bg_image

                img_sum = np.sum(img[154:454, 264:644])
                #img_sum = np.sum(img)
                cpc = (img_charge * num_of_pulses) / (img_sum * magic_const * in_filter)

                mask_dict.setdefault(delay_slots, []).append(run_counter)

                first_images.setdefault(run, {})
                if delay_slots not in first_images[run]:
                    first_images[run][delay_slots] = (img.copy(), delay_slots)


                run_cpcs.append(cpc)
                run_counter += 1

        run_events = list(range(run_counter))
        run_results = {'cpcs': run_cpcs, 'events': run_events}

        for delay, indices in mask_dict.items():
            run_results[f'mask_{delay}'] = indices

        results[run] = run_results
        print("Events:", len(run_events))

    return results, first_images


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


if __name__ == "__main__":
    runs = ['run_02', 'run_04', 'run_09', 'run_11', 'run_13']
    japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')

    dctx = zstd.ZstdDecompressor()
    directory = '/user/jpucek/BSRH/Data/2025/06/20/'
    files = sorted(os.listdir(directory))

    results, first_images = compute_cpc(directory, files, runs)
    graph_cpc(results)

    plot_example_images(first_images)
