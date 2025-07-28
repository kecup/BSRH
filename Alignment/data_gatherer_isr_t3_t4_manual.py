import pyjapc
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from devices_lib import Camera, Motor, gaussian
from AlignToTarget import Aligner, FeedbackNN, BeamSteerer
import torch
import numpy as np
from time import sleep
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

MOTOR_REFERENCES = {"x": 40085, "y": 41740, "phi": 141900, "theta": 161000}
target = [484, 304]  # Camera center

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

def main():

    model = load_model()

    japc.startSubscriptions()

    motor_x.setRelPosition(30)
    motor_y.setRelPosition(30)
    motor_phi.setRelPosition(30)
    motor_theta.setRelPosition(30)
    sleep(2)

    while True:
        cam_pos = np.array(cam2.getBeamPosition(), dtype=np.float32)
        if (cam_pos[0] == -1 or cam_pos[1] == -1):
            print("beam lost")
            print("Sending to references.")
            move_to_references()
        input_vector = np.concatenate((cam_pos, target), axis=0, dtype=np.float32())
        input_tensor = torch.tensor(input_vector).unsqueeze(0)
        with torch.no_grad():
            prediction = model(input_tensor).squeeze(0).numpy()  # steps
        motor_x.setRelPosition(int(prediction[0]))  # move to prediction
        motor_y.setRelPosition(int(prediction[1]))  # move to prediction
        while motor_x.isBusy() or motor_y.isBusy() or motor_phi.isBusy() or motor_theta.isBusy():
            sleep(0.2)

        sleep(2)

    japc.stopSubscriptions()
    japc.clearSubscriptions()


if __name__ == "__main__":
    main()