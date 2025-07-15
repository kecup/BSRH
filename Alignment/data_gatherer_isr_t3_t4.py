# gather data on isr - mirror2 (x, y, phi, theta) + target3 (see-through) + target4

import pyjapc
from time import sleep
import csv
import random
import sys
import os
import math

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from devices_lib import Camera, Motor, Pneumatics
import pyrbac


def main():
    japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
    client = pyrbac.AuthenticationClient()
    token = client.login_kerberos()
    japc.rbacLogin(token=token)

    # ISR Motors
    motor_x = Motor('TST.cfv-375-bsrdev.mot.3-2', japc, 5)
    motor_y = Motor('TST.cfv-375-bsrdev.mot.3-3', japc, 5)
    motor_phi = Motor('TST.cfv-375-bsrdev.mot.3-4', japc, 168)
    motor_theta = Motor('TST.cfv-375-bsrdev.mot.3-5', japc, 125)

    # ISR Cameras
    cam1 = Camera("TST.BSRH.MON.M2", japc)
    cam2 = Camera("TST.BSRH.CAM.CM", japc)

    # How many steps the motor can make from reference to keep the image on the camera
    # we only care about the second camera
    motor_range_x_t2 = 320
    motor_range_y_t2 = 620
    motor_range_phi_t2 = 450
    motor_range_theta_t2 = 900

    japc.startSubscriptions()
    sleep(2)

    # reference position to return to if something goes wrong
    ref_x = motor_x.getPosition() # 40085
    ref_y = motor_y.getPosition() # 41740
    ref_phi = motor_phi.getPosition() # 163500
    ref_theta = motor_theta.getPosition() # 140600

    iterations = 20


    # open a csv file that we will use to save the training data:
    file_name = 'isr_t3_t4_data.csv'
    header = ["cam1X", "cam1Y", "cam2X", "cam2Y", "mX", "mY", "mPhi", "mTheta", "t1X", "t1Y", "t2X", "t2Y", "stepsX",
              "stepsY", "stepsPhi", "stepsTheta"]
    file_exists = os.path.isfile(file_name)

    with open(file_name, 'a') as file:
        file_writer = csv.writer(file)

        if not file_exists:
            file_writer.writerow(header)

        # collect data
        # file_writer.writerow([])
        # flush after every append: file.flush()
        for i in range(iterations):
            print("Starting iteration " + str(i + 1))

            cam1X, cam1Y = measure_cam1()

            mX = motor_x.getPosition()
            mY = motor_y.getPosition()
            mPhi = motor_phi.getPosition()
            mTheta = motor_theta.getPosition()

            cam2X, cam2Y = measure_cam2()

            stepsX, stepsY, stepsPhi, stepsTheta = get_steps()
            move_motors(stepsX, stepsY, stepsPhi, stepsTheta)

            t1X, t1Y = cam1.getBeamPosition()
            t2X, t2Y = cam2.getBeamPosition()

            if not math.isclose(t1X, -1.0, rel_tol=1e-9) and not math.isclose(t1Y, -1.0,rel_tol=1e-9) and not math.isclose(t2X,-1.0,
                                            rel_tol=1e-9) and not math.isclose(t2Y, -1.0, rel_tol=1e-9):
                row = [cam1X, cam1Y, cam2X, cam2Y, mX, mY, mPhi, mTheta, t1X, t1Y, t2X, t2Y, stepsX, stepsY, stepsPhi, stepsTheta]
                file_writer.writerow(row)
                file.flush()



    motor_x.setAbsPosition(ref_x)
    motor_y.setAbsPosition(ref_y)
    motor_phi.setAbsPosition(ref_phi)
    motor_theta.setAbsPosition(ref_theta)

    japc.stopSubscriptions()

if __name__ == "__main__":
    main()