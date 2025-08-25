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
motor_x = Motor('TST.cfv-375-bsrdev.mot.3-2', japc, 5)
motor_y = Motor('TST.cfv-375-bsrdev.mot.3-3', japc, 5)
motor_phi = Motor('TST.cfv-375-bsrdev.mot.3-4', japc, 168)
motor_theta = Motor('TST.cfv-375-bsrdev.mot.3-5', japc, 125)

cam2 = Camera("TST.BSRH.CAM.CM", japc)

MOTOR_REFERENCES = {"x": 40385, "y": 41740, "phi": 140381, "theta": 163301}

motor_range_phi_plus_t2 = 900
motor_range_phi_minus_t2 = 1000
motor_range_theta_plus_t2 = 500
motor_range_theta_minus_t2 = 650


def get_steps_only_phi_theta():
    phi_sign = random.randint(0, 1)
    if phi_sign == 1:
        steps_phi = (random.randint(MOTOR_REFERENCES["phi"] - motor_range_phi_plus_t2,
                                    MOTOR_REFERENCES["phi"] + motor_range_phi_plus_t2)) - motor_phi.getPosition()
    else:
        steps_phi = (random.randint(MOTOR_REFERENCES["phi"] - motor_range_phi_minus_t2,
                                    MOTOR_REFERENCES["phi"] + motor_range_phi_minus_t2)) - motor_phi.getPosition()

    theta_sign = random.randint(0, 1)
    if theta_sign == 1:

        steps_theta = (random.randint(MOTOR_REFERENCES["theta"] - motor_range_theta_plus_t2,
                                      MOTOR_REFERENCES["theta"] + motor_range_theta_plus_t2)) - motor_theta.getPosition()
    else:
        steps_theta = (random.randint(MOTOR_REFERENCES["theta"] - motor_range_theta_minus_t2,
                                      MOTOR_REFERENCES["theta"] + motor_range_theta_minus_t2)) - motor_theta.getPosition()

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

def main():
    japc.startSubscriptions()
    sleep(2)
    move_motors(0,0, 2000,0)
    sleep(1)
    cam2.captureImage(True)

    move_to_references()
    sleep(1)

    file_name = "Data/test_center_of_mass.csv"
    header = ["gaussX", "gaussY", "comX", "comY", "stepsPhi", "stepsTheta"] # com = center of mass
    file_exists = os.path.isfile(file_name)

    with (open(file_name, 'a') as file):
        file_writer = csv.writer(file)

        if not file_exists:
            file_writer.writerow(header)

        for i in range(500):
            sleep(0.3) # pro jistotu
            gauss_pos = cam2.getBeamPosition()
            com_pos = cam2.getBeamPosition(com=True)

            stepsX, stepsY, stepsPhi, stepsTheta = get_steps_only_phi_theta()
            move_motors(stepsX, stepsY, stepsPhi, stepsTheta)

            if not math.isclose(gauss_pos[0], -1.0, rel_tol=1e-9) and not math.isclose(gauss_pos[1], -1.0,rel_tol=1e-9) and not math.isclose(com_pos[0],-1.0, rel_tol=1e-9) and not math.isclose(com_pos[1], -1.0, rel_tol=1e-9):

                row = [gauss_pos[0], gauss_pos[1], com_pos[0], com_pos[1], stepsPhi, stepsTheta]
                file_writer.writerow(row)
                file.flush()
            else:
                print("Invalid targets.")
                move_to_references()
                print("Sending to references.")

    move_to_references()

    japc.stopSubscriptions()


if __name__ == "__main__":
    main()