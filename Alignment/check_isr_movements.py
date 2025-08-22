import pyjapc
import sys
import os
from time import sleep
import csv
from datetime import datetime
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import pyrbac
from devices_lib import Camera

def main():
    japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
    client = pyrbac.AuthenticationClient()
    token = client.login_kerberos()
    japc.rbacLogin(token=token)

    cam = Camera("TST.BSRH.CAM.CM", japc)

    japc.startSubscriptions()
    sleep(2)

    file_name = 'Data/isr_motor_movements.csv'
    header = ["cam1X", "cam1Y", "time"]
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


    japc.stopSubscriptions()


if __name__ == "__main__":
    main()