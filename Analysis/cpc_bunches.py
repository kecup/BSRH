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
    for file_idx in range(len(files)):
        if run in files[file_idx]:
            f = open(directory + files[file_idx], 'rb')
            with dctx.stream_reader(f) as reader:
                file_data = pickle.load(reader)
                data.append(file_data)
    return data


def graph_cpc(data):

    n_runs = len(data)

    # Determine subplot grid size:
    n_cols = 2
    n_rows = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()  # flatten in case of 2D array even if 1D needed

    colors = ["orange", "green", "skyblue", "purple", "pink"]
    labels  = ["bunch 20", "bunch 500", "bunch 1050", "bunch 1250", "bunch 1800"]

    for idx, (run_num, run_data) in enumerate(data.items()):
        ax = axes[idx]
        events = run_data['events']
        cpc = run_data['cpcs']

        for mask_idx, mask_name in enumerate(['mask_0', 'mask_40', 'mask_80', 'mask_120', 'mask_160']):
            mask = run_data[mask_name]
            ax.scatter(
                [events[i] for i in mask],  # x-axis values (events)
                [cpc[i] for i in mask],  # y-axis values (cpc)
                color=colors[mask_idx],
                label=labels[mask_idx],
                s=12
            )

        ax.set_title(f'{run_num}')
        ax.set_xlabel('Events')
        ax.set_ylabel('CPC')
        ax.legend()

    # Hide unused subplots:
    for i in range(n_runs, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def compute_cpc(directory: str, files, runs: list[str]):
    """
    Computes charge per count for every image for every event in a run.

    :param runs: list of run number to iterate over
    """
    results = {}
    for i in range(len(runs)):
        run = runs[i]
        print("Analysing " + run)
        run_data = get_run_data(directory, files, run)
        mask_0 = []
        mask_40 = []
        mask_80 = []
        mask_120 = []
        mask_160 = []
        run_cpcs = []
        run_counter = 0
        bg_image, bg_pulses = get_background(directory, files[36], 50)
        for file_num in range(len(run_data)):
            a = run_data[file_num]
            for event_idx in range(len(a)):
                event = a[event_idx]
                img = event.cam_data['image2D']
                delay_slots = int(event.device_data["GATE-INT"]["delaySlots"])
                pulse_length = int(event.device_data["GATE-INT"]["pulseLength"])
                img_charge = np.sum(event.device_data["FBCT1"]["averageBunchIntensities"][delay_slots:delay_slots + pulse_length // 4][0])

                fArr = event.device_data['FILTERS.B2']['channelState']
                in_filter = pow(12.6, 3 * fArr[0] + 2 * fArr[1] + fArr[2])
                magic_const = 5651
                num_of_pulses = int(event.device_data["GATE-INT"]["numberOfPulses"])
                #img = img - (num_of_pulses / bg_pulses) * bg_image
                img_sum = np.sum(img[154:454, 264:644])
                cpc = (img_charge * num_of_pulses) / (img_sum * magic_const * in_filter)

                if delay_slots == 20:
                    mask_0.append(run_counter)
                elif delay_slots == 500:
                    mask_40.append(run_counter)
                elif delay_slots == 1050:
                    mask_80.append(run_counter)
                elif delay_slots == 1250:
                    mask_120.append(run_counter)
                elif delay_slots == 1800:
                    mask_160.append(run_counter)
                elif delay_slots == 1800:
                    mask_160.append(run_counter)
                run_cpcs.append(cpc)
                run_counter += 1
        run_events = list(range(run_counter))
        run_results = {'cpcs': run_cpcs, 'events': run_events, 'mask_0': mask_0, 'mask_40': mask_40, 'mask_80': mask_80, 'mask_120': mask_120, 'mask_160': mask_160}
        results[run] = run_results
        print("Events: " + str(len(run_events)))
    return results

if __name__ == "__main__":
    runs = ['run_02', 'run_04', 'run_09', 'run_11', 'run_13']
    japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')  # Stupid Java VM must be running to open the java values... We have to retype this...

    dctx = zstd.ZstdDecompressor()
    directory = '/user/jpucek/BSRH/Data/2025/06/20/'
    files = os.listdir(directory)
    files = sorted(files)

    results = compute_cpc(directory, files, runs)
    graph_cpc(results)

