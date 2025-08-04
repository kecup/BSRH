import pyjapc
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from devices_lib import Camera, Motor
from AlignToTarget import Aligner_without_beam_steerer, FeedbackNN
import torch
import numpy as np
from time import sleep
import csv
import matplotlib.pyplot as plt
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
CSV_FILENAME = "Data/isr_t3_t4_auto_minusY_data.csv"
TARGET = [484, 304]  # Camera center

def load_model():
    """
    Loads the best current model for 1 camera and two motors (x, y).
    cam + target -> steps
    """
    file_name4 = "Models/model2_cam_target-steps.pth"
    model4 = FeedbackNN(4, 2)
    model4.load_state_dict(torch.load(file_name4))
    model4.eval()
    return model4

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

def calculate_distance(pos, target):
    """
    Calculates distance of current position to target position.
    """
    return np.linalg.norm(np.array(pos) - np.array(target))

def reverse_motors(stepsX, stepsY, stepsPhi, stepsTheta):
    """
    Reverses motor moves.
    """
    motor_x.setRelPosition(-1*stepsX)
    motor_y.setRelPosition(-1*stepsY)
    motor_phi.setRelPosition(-1*stepsPhi)
    motor_theta.setRelPosition(-1*stepsTheta)
    while motor_x.isBusy() or motor_y.isBusy() or motor_phi.isBusy() or motor_theta.isBusy():
        sleep(0.2)

def manual_collection(aligner, file_writer, align_iter=3):
    while True:
        cam1_pos = cam1.getBeamPosition()
        cam2_pos = cam2.getBeamPosition()
        deltas = input(
            "Move motors and enter two integers separated by space as in delta steps X (3.2) and delta steps Y (3.3): ")
        try:
            stepsX, stepsY = map(int, deltas.split())  # Convert to integers
            print(f"You entered: {stepsX} and {stepsY}")
        except ValueError:
            print("Invalid input. Please enter two integers.")
            continue
        print("auto-aligning")
        stepsPhi, stepsTheta = aligner.align(align_iter)
        t1 = cam1.getBeamPosition()
        t2 = cam2.getBeamPosition()

        if t1[0] != -1 and t1[1] != -1 and t2[0] != -1 and t2[1] != -1:
            data = [cam1_pos[0], cam1_pos[1], cam2_pos[0], cam2_pos[1], t1[0], t1[1], t2[0], t2[1], stepsX, stepsY,
                    stepsPhi, stepsTheta]
            file_writer.writerow(data)
        else:
            print("beam lost")
            reverse_motors(stepsX, stepsY, stepsPhi, stepsTheta)

def auto_collection(aligner, file_writer, align_iter=3, stepsX=0, stepsY=0):
    i = 1
    while True:
        print(f"iteration {i}")
        cam1_pos = cam1.getBeamPosition()
        cam2_pos = cam2.getBeamPosition()

        motor_x.setRelPosition(stepsX) # move motors
        motor_y.setRelPosition(stepsY)
        while motor_x.isBusy() or motor_y.isBusy():
            sleep(0.2)

        if motor_x.limitSwitchActive() or motor_y.limitSwitchActive(): # stopping condition 1
            print("E0: limit switch hit, stopping")
            break

        cam2_check_pos = cam2.getBeamPosition()
        if cam2_check_pos[0] == -1 or cam2_check_pos[1] == -1:
            print("E1: beam lost, moved too far")
            break

        stepsPhi, stepsTheta = aligner.align(align_iter) # auto-align
        t1 = cam1.getBeamPosition()
        t2 = cam2.getBeamPosition()

        if t1[0] != -1 and t1[1] != -1 and t2[0] != -1 and t2[1] != -1: # stopping condition 2
            data = [cam1_pos[0], cam1_pos[1], cam2_pos[0], cam2_pos[1], t1[0], t1[1], t2[0], t2[1], stepsX, stepsY,
                    stepsPhi, stepsTheta]
            file_writer.writerow(data)
        else:
            print("E2: beam lost, stopping")
            break
        i += 1


def main():

    model = load_model()
    aligner = Aligner_without_beam_steerer(motor_phi, motor_theta, cam2, TARGET, model, MOTOR_REFERENCES["phi"], MOTOR_REFERENCES["theta"])

    japc.startSubscriptions()

    move_to_references()
    sleep(1)

    header = ["cam1X", "cam1Y", "cam2X", "cam2Y", "t1X", "t1Y", "t2X", "t2Y", "stepsX",
              "stepsY", "stepsPhi", "stepsTheta"]
    file_exists = os.path.isfile(CSV_FILENAME)

    with open(CSV_FILENAME, 'a') as file:
        file_writer = csv.writer(file)
        if not file_exists:
            file_writer.writerow(header)

        aligner.align(5) # init align

        # manual_collection(aligner, file_writer, align_iter=2)
        auto_collection(aligner, file_writer, align_iter=2, stepsX=0, stepsY=-300)


    japc.stopSubscriptions()
    japc.clearSubscriptions()


if __name__ == "__main__":
    main()