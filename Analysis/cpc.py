import dill as pickle
import pyjapc
import zstandard as zstd
import numpy as np
import os
import matplotlib.pyplot as plt


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
    n_cols = 3  # e.g. 2 plots per row
    # n_rows = (n_runs + n_cols - 1) // n_cols  # ceiling division to get enough rows
    n_rows = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()  # flatten in case of 2D array even if 1D needed

    for idx, (run_num, run_data) in enumerate(data.items()):
        ax = axes[idx]
        ax.plot(run_data['events'], run_data['cpcs'], marker='o', linestyle='-')
        ax.set_title(f'Run {run_num}')
        ax.set_xlabel("Events")
        ax.set_ylabel("cpc")

    # Hide unused subplots if any:
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
        print("Running " + run)
        run_data = get_run_data(directory, files, run)
        run_cpcs = []
        events_counter = 0
        for file_num in range(len(run_data)):
            a = run_data[file_num]
            events_counter += len(a) # this is prob wrong
            for event_idx in range(len(a)):
                event = a[event_idx]
                img = event.cam_data['image2D']
                delay_slots = int(event.device_data["GATE-INT"]["delaySlots"])
                pulse_length = int(event.device_data["GATE-INT"]["pulseLength"])
                img_charge = np.sum(event.device_data["FBCT1"]["averageBunchIntensities"][delay_slots:delay_slots + pulse_length // 4][0])
                img_sum = np.sum(img)
                in_filter = 1
                magic_const = 5651
                num_of_pulses = int(event.device_data["GATE-INT"]["numberOfPulses"])
                cpc = (img_charge * num_of_pulses) / (img_sum * magic_const * in_filter)
                run_cpcs.append(cpc)
        run_events = list(range(events_counter))
        run_results = {'cpcs': run_cpcs, 'events': run_events}
        results[run] = run_results
        # print(run_results)
        # print(len(run_cpcs))
        print("Events: " + str(len(run_events)))
    return results

if __name__ == "__main__":
    runs = ['run_06', 'run_07', 'run_08', 'run_09', 'run_10', 'run_11', 'run_12']
    japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')  # Stupid Java VM must be running to open the java values... We have to retype this...

    dctx = zstd.ZstdDecompressor()
    directory = '/user/jpucek/BSRH/Data/2025/06/19/'
    files = os.listdir(directory)
    files = sorted(files)

    results = compute_cpc(directory, files, runs)
    # print(results)
    graph_cpc(results)

