import pyjapc
import sys
import os
from time import sleep
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import matplotlib.dates as mdates
from datetime import datetime, timedelta
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import pyrbac
from devices_lib import Camera

def collect_data(cam, file_name):
    file_name = file_name
    header = ["gaussX", "gaussY", "comX", "comY", "time"]
    file_exists = os.path.isfile(file_name)

    now = datetime.now()
    # Target time = tomorrow 10:00 AM
    tomorrow = now.date() + timedelta(days=2)
    target_time = datetime.combine(tomorrow, datetime.strptime("10:00", "%H:%M").time())

    with open(file_name, 'a') as file:
        file_writer = csv.writer(file)

        if not file_exists:
            file_writer.writerow(header)

        while True:
            cam_pos = cam.getBeamPosition()
            com_pos = cam.getBeamPosition(True)
            time = datetime.now()
            if now > target_time:
                print("It's later than tomorrow 10:00. Stopping script.")
                file.flush()
                sys.exit()

            row = [cam_pos[0], cam_pos[1], com_pos[0], com_pos[1], time]
            file_writer.writerow(row)
            file.flush()
            sleep(10)

def visualize_data(file_name):
    df = pd.read_csv(file_name)

    print(f"There are {df.shape[0]} rows")

    df["time"] = pd.to_datetime(df["time"])

    # Convert to numeric for coloring
    time_numeric = df["time"].map(mdates.date2num)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df["gaussX"], df["gaussY"], c=time_numeric, cmap="viridis", s=10)
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

    # ISR
    # cam = Camera("TST.BSRH.CAM.CM", japc) # core mask
    # cam = Camera("TST.BSRH.MON.M2", japc) # t3
    # cam = Camera("TST.BSRH.CAM.MAIN", japc) # Main camera

    # LHC
    cam = Camera("LHC.BSRH.CAM.CM.B2", japc)

    file_name = 'Data/lhc_motor_movements_cm.csv'

    japc.startSubscriptions()
    sleep(2)

    # collect_data(cam, file_name)
    visualize_data(file_name)

    japc.stopSubscriptions()


if __name__ == "__main__":
    main()