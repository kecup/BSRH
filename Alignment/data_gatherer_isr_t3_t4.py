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

MOTOR_REFERENCES = {"x": 40385, "y": 41740, "phi": 139931, "theta": 163051}

# How many steps the motor can make from reference to keep the image on the camera
# we only care about the second camera
motor_range_x_plus_t2 = 350
motor_range_x_minus_t2 = 350
motor_range_y_plus_t2 = 550
motor_range_y_minus_t2 = 700
motor_range_phi_plus_t2 = 1050
motor_range_phi_minus_t2 = 800
motor_range_theta_plus_t2 = 600
motor_range_theta_minus_t2 = 500


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
    x_sign = random.randint(0, 1)
    if x_sign == 1:
        steps_x = (random.randint(motor_x.getPosition() - motor_range_x_plus_t2,
                                  motor_x.getPosition() + motor_range_x_plus_t2)) - motor_x.getPosition()
    else:
        steps_x = (random.randint(motor_x.getPosition() - motor_range_x_minus_t2,
                                  motor_x.getPosition() + motor_range_x_minus_t2)) - motor_x.getPosition()

    y_sign = random.randint(0,1)
    if y_sign == 1:

        steps_y = (random.randint(motor_y.getPosition() - motor_range_y_plus_t2,
                                      motor_y.getPosition() + motor_range_y_plus_t2)) - motor_y.getPosition()
    else:
        steps_y = (random.randint(motor_y.getPosition() - motor_range_y_minus_t2,
                                      motor_y.getPosition() + motor_range_y_minus_t2)) - motor_y.getPosition()

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

    return steps_x, steps_y, steps_phi, steps_theta

def get_steps_only_x_y():
    x_sign = random.randint(0,1)
    if x_sign == 1:
        steps_x = (random.randint(motor_x.getPosition() - motor_range_x_plus_t2,
                                  motor_x.getPosition() + motor_range_x_plus_t2)) - motor_x.getPosition()
    else:
        steps_x = (random.randint(motor_x.getPosition() - motor_range_x_minus_t2,
                                  motor_x.getPosition() + motor_range_x_minus_t2)) - motor_x.getPosition()

    y_sign = random.randint(0,1)
    if y_sign == 1:

        steps_y = (random.randint(motor_y.getPosition() - motor_range_y_plus_t2,
                                      motor_y.getPosition() + motor_range_y_plus_t2)) - motor_y.getPosition()
    else:
        steps_y = (random.randint(motor_y.getPosition() - motor_range_y_minus_t2,
                                      motor_y.getPosition() + motor_range_y_minus_t2)) - motor_y.getPosition()

    return steps_x, steps_y, 0, 0

def get_steps_only_phi_theta():
    phi_sign = random.randint(0,1)
    if phi_sign == 1:
        steps_phi = (random.randint(motor_phi.getPosition() - motor_range_phi_plus_t2,
                                  motor_phi.getPosition() + motor_range_phi_plus_t2)) - motor_phi.getPosition()
    else:
        steps_phi = (random.randint(motor_phi.getPosition() - motor_range_phi_minus_t2,
                                  motor_phi.getPosition() + motor_range_phi_minus_t2)) - motor_phi.getPosition()

    theta_sign = random.randint(0,1)
    if theta_sign == 1:

        steps_theta = (random.randint(motor_theta.getPosition() - motor_range_theta_plus_t2,
                                      motor_theta.getPosition() + motor_range_theta_plus_t2)) - motor_theta.getPosition()
    else:
        steps_theta = (random.randint(motor_theta.getPosition() - motor_range_theta_minus_t2,
                                      motor_theta.getPosition() + motor_range_theta_minus_t2)) - motor_theta.getPosition()

    return 0, 0, steps_phi, steps_theta

def move_motors(stepsX, stepsY, stepsPhi, stepsTheta):
    motor_x.setRelPosition(stepsX)
    motor_y.setRelPosition(stepsY)
    motor_phi.setRelPosition(stepsPhi)
    motor_theta.setRelPosition(stepsTheta)
    while motor_x.isBusy() or motor_y.isBusy() or motor_phi.isBusy() or motor_theta.isBusy():
        sleep(0.2)

def move_to_references():
    """
    Moves motors to pre-measured references.
    """
    motor_x.setAbsPosition(MOTOR_REFERENCES["x"])
    motor_y.setAbsPosition(MOTOR_REFERENCES["y"])
    motor_phi.setAbsPosition(MOTOR_REFERENCES["phi"])
    motor_theta.setAbsPosition(MOTOR_REFERENCES["theta"])
    while motor_x.isBusy() or motor_y.isBusy() or motor_phi.isBusy() or motor_theta.isBusy():
        sleep(0.2)

def move_motors_by_backlash():
    motor_x.setRelPosition(motor_x.backlash_steps)
    motor_y.setRelPosition(motor_y.backlash_steps)
    motor_phi.setRelPosition(motor_phi.backlash_steps)
    motor_theta.setRelPosition(motor_theta.backlash_steps)
    while motor_x.isBusy() or motor_y.isBusy() or motor_phi.isBusy() or motor_theta.isBusy():
        sleep(0.2)

def main():

    japc.startSubscriptions()
    sleep(2)

    move_motors_by_backlash()
    move_to_references()

    iterations = 50000
    faults = 0

    # open a csv file that we will use to save the training data:
    file_name = "Data/isr_t3_t4_data_v3.csv"
    header = ["cam1X", "cam1Y", "cam2X", "cam2Y", "mX", "mY", "mPhi", "mTheta", "t1X", "t1Y", "t2X", "t2Y", "stepsX",
              "stepsY", "stepsPhi", "stepsTheta"]
    file_exists = os.path.isfile(file_name)

    with open(file_name, 'a') as file:
        file_writer = csv.writer(file)

        if not file_exists:
            file_writer.writerow(header)

        # collect data
        for i in range(iterations):
            print("Starting iteration " + str(i + 1) + " out of " + str(iterations))
            print("Faults so far: " + str(faults))

            cam1X, cam1Y = measure_cam(cam1)
            cam2X, cam2Y = measure_cam(cam2)

            mX = motor_x.getPosition()
            mY = motor_y.getPosition()
            mPhi = motor_phi.getPosition()
            mTheta = motor_theta.getPosition()

            stepsX, stepsY, stepsPhi, stepsTheta = get_steps()
            # stepsX, stepsY, stepsPhi, stepsTheta = get_steps_only_phi_theta()
            # print(stepsX, stepsY, stepsPhi, stepsTheta)
            move_motors(stepsX, stepsY, stepsPhi, stepsTheta)

            t1X, t1Y = cam1.getBeamPosition()
            t2X, t2Y = cam2.getBeamPosition()

            if not math.isclose(t1X, -1.0, rel_tol=1e-9) and not math.isclose(t1Y, -1.0,rel_tol=1e-9) and not math.isclose(t2X,-1.0,
                                            rel_tol=1e-9) and not math.isclose(t2Y, -1.0, rel_tol=1e-9):
                row = [cam1X, cam1Y, cam2X, cam2Y, mX, mY, mPhi, mTheta, t1X, t1Y, t2X, t2Y, stepsX, stepsY, stepsPhi, stepsTheta]
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