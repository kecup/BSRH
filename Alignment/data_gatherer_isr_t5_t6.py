# gather data on isr - mirror3 (phi, theta) + target5 (Lyot Stop) + target6 (MAIN CAM)

import pyjapc
from time import sleep
import csv
import random
import sys
import os
import math

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from devices_lib import Camera, Motor
import pyrbac

japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
client = pyrbac.AuthenticationClient()
token = client.login_kerberos()
japc.rbacLogin(token=token)

# ISR Motors
motor_phi = Motor('TST.cfv-375-bsrdev.mot.5-1', japc, 11)
motor_theta = Motor('TST.cfv-375-bsrdev.mot.5-2', japc, 8)

# ISR Cameras
cam1 = Camera("TST.BSRH.MON.LS", japc) # Lyot stop
cam2 = Camera("TST.BSRH.CAM.MAIN", japc) # Main camera

MOTOR_REFERENCES = {"phi": 13444, "theta": 13360}

# How many steps the motor can make from reference to keep the image on the camera
motor_range_phi_plus_t2 = 360
motor_range_phi_minus_t2 = 360
motor_range_theta_plus_t2 = 300
motor_range_theta_minus_t2 = 300


def measure_cam(camera):
    """
    Measure camera data.
    """
    camX, camY = camera.getBeamPosition()
    if camX == -1 or camY == -1:
        print('Did not find beam, sending to references.')
        move_to_references()
        camX, camY = camera.getBeamPosition()
        if camX == -1 or camY == -1:
            print('Cannot find beam at reference location. This should not happen.')
            # something is wrong, this shouldn't happen
    return camX, camY

def get_steps():
    """
    Randomly generate motor movements based on their range and current position.
    """
    phi_sign = random.randint(0, 1)
    if phi_sign == 1:
        steps_phi = (random.randint(motor_phi.getPosition() - motor_range_phi_plus_t2,
                                    motor_phi.getPosition() + motor_range_phi_plus_t2)) - motor_phi.getPosition()
    else:
        steps_phi = (random.randint(motor_phi.getPosition() - motor_range_phi_minus_t2,
                                    motor_phi.getPosition() + motor_range_phi_minus_t2)) - motor_phi.getPosition()

    theta_sign = random.randint(0, 1)
    if theta_sign == 1:

        steps_theta = (random.randint(motor_theta.getPosition() - motor_range_theta_plus_t2,
                                      motor_theta.getPosition() + motor_range_theta_plus_t2)) - motor_theta.getPosition()
    else:
        steps_theta = (random.randint(motor_theta.getPosition() - motor_range_theta_minus_t2,
                                      motor_theta.getPosition() + motor_range_theta_minus_t2)) - motor_theta.getPosition()

    return steps_phi, steps_theta


def move_motors(stepsPhi, stepsTheta):
    """
    Move motors by given steps.
    """
    motor_phi.setRelPosition(stepsPhi)
    motor_theta.setRelPosition(stepsTheta)
    while motor_phi.isBusy() or motor_theta.isBusy():
        sleep(0.2)

def move_to_references():
    """
    Move motors to pre-measured references.
    """
    motor_phi.setAbsPosition(MOTOR_REFERENCES["phi"])
    motor_theta.setAbsPosition(MOTOR_REFERENCES["theta"])
    while motor_phi.isBusy() or motor_theta.isBusy():
        sleep(0.2)

def move_motors_by_backlash():
    """
    Move motors by their backlash.
    """
    motor_phi.setRelPosition(motor_phi.backlash_steps)
    motor_theta.setRelPosition(motor_theta.backlash_steps)
    while motor_phi.isBusy() or motor_theta.isBusy():
        sleep(0.2)

def main():

    japc.startSubscriptions()
    sleep(2)

    move_motors_by_backlash()
    move_to_references()

    iterations = 25000
    faults = 0

    # open a csv file that we will use to save the training data:
    file_name = "Data/isr_t5_t6_data_v1.csv"
    header = ["cam2X", "cam2Y", "mPhi", "mTheta", "t2X", "t2Y", "stepsPhi", "stepsTheta"]
    file_exists = os.path.isfile(file_name)

    with open(file_name, 'a') as file:
        file_writer = csv.writer(file)

        if not file_exists:
            file_writer.writerow(header)

        # collect data
        for i in range(iterations):
            print("Starting iteration " + str(i + 1) + " out of " + str(iterations))
            print("Faults so far: " + str(faults))

            cam2X, cam2Y = measure_cam(cam2)

            mPhi = motor_phi.getPosition()
            mTheta = motor_theta.getPosition()

            stepsPhi, stepsTheta = get_steps()
            move_motors(stepsPhi, stepsTheta)

            t2X, t2Y = cam2.getBeamPosition()

            if not math.isclose(t2X,-1.0, rel_tol=1e-9) and not math.isclose(t2Y, -1.0, rel_tol=1e-9):
                row = [cam2X, cam2Y, mPhi, mTheta, t2X, t2Y, stepsPhi, stepsTheta]
                file_writer.writerow(row)
                file.flush()
            else:
                print("Invalid targets.")
                faults += 1
                move_to_references()
                print("Sending to references.")


    move_to_references()

    japc.stopSubscriptions()

if __name__ == "__main__":
    main()