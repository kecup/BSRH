import pyjapc
import sys
import os
from time import sleep
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import matplotlib.dates as mdates
from datetime import datetime
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import pyrbac
from devices_lib import Camera

def collect_data(cam):
    file_name = 'Data/isr_motor_movements_2.csv'
    header = ["camX", "camY", "time"]
    file_exists = os.path.isfile(file_name)

    with open(file_name, 'a') as file:
        file_writer = csv.writer(file)

        if not file_exists:
            file_writer.writerow(header)

        while True:
            cam_pos = cam.getBeamPosition()
            time = datetime.now()
            row = [cam_pos[0], cam_pos[1], time]
            file_writer.writerow(row)
            file.flush()
            sleep(10)

def visualize_data():
    df = pd.read_csv('Data/isr_motor_movements_2.csv')

    print(f"There are {df.shape[0]} rows")

    df["time"] = pd.to_datetime(df["time"])

    # Convert to numeric for coloring
    time_numeric = df["time"].map(mdates.date2num)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df["camX"], df["camY"], c=time_numeric, cmap="viridis", s=10)
    plt.xlabel("x coordinate")
    plt.ylabel("y coordinate")
    plt.title("Trajectory colored by Time")

    # Colorbar with proper datetime labels
    cbar = plt.colorbar(scatter)
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    cbar.set_label("Time")
    plt.show()


def main():
    japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
    client = pyrbac.AuthenticationClient()
    token = client.login_kerberos()
    japc.rbacLogin(token=token)

    cam = Camera("TST.BSRH.CAM.CM", japc)

    japc.startSubscriptions()
    sleep(2)

    # collect_data(cam)
    visualize_data()

    japc.stopSubscriptions()


if __name__ == "__main__":
    main()