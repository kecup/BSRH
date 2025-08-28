import pyjapc
import sys
import os
import pyrbac
import numpy as np
import torch
from time import sleep
import joblib

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from devices_lib import Camera, Motor
from TrainNN import FeedbackNN6

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

MOTOR_REFERENCES = {"x": 40385, "y": 41740, "phi": 140381, "theta": 163301}
CAM1_TARGET = [751, 144] # cam1 target centre


def generate_targets():
    targets = [[145, 466], [135, 135], [800, 172], [814, 480], [106, 290], [477, 113], [801, 313], [470, 483],
               [282, 434], [775, 205], [697, 420], [308, 238], [484, 304]]
    return targets

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

def move_motors(stepsX, stepsY, stepsPhi, stepsTheta):
    motor_x.setRelPosition(stepsX)
    motor_y.setRelPosition(stepsY)
    motor_phi.setRelPosition(stepsPhi)
    motor_theta.setRelPosition(stepsTheta)
    while motor_x.isBusy() or motor_y.isBusy() or motor_phi.isBusy() or motor_theta.isBusy():
        sleep(0.2)


def move_to_target(model, targets_cam1, X_scaler, y_scaler):
    counts = []
    faults = 0
    for i in range(len(targets_cam1)):
        target = np.array(targets_cam1[i], dtype=np.float32)
        count = 0
        cam1_pos = np.array(cam1.getBeamPosition(), dtype=np.float32)
        cam2_pos = np.array(cam2.getBeamPosition(), dtype=np.float32)
        while cam2_pos[0] == -1 or cam2_pos[1] == -1:
            if count > 15:
                print("Not getting closer reasonably fast.")
                faults += 1
                count = -2
                break
            if (cam1_pos[0] == -1 or cam1_pos[1] == -1):
                print("beam lost")
                print("Sending to references.")
                count = -1
                break

            input_vector = np.concatenate((cam1_pos, target), axis=0).astype(np.float32)
            input_vector_2d = input_vector.reshape(1, -1)
            scaled_input = X_scaler.transform(input_vector_2d)
            scaled_tensor = torch.from_numpy(scaled_input).float()

            with torch.no_grad():
                prediction_scaled = model(scaled_tensor).numpy()

            prediction_real = y_scaler.inverse_transform(prediction_scaled)
            move_motors(0, 0, prediction_real[0][0], prediction_real[0][1]) # only phi + theta moves

            cam1_pos = np.array(cam1.getBeamPosition(), dtype=np.float32)
            cam2_pos = np.array(cam2.getBeamPosition(), dtype=np.float32) # new camera position
            count += 1
        print("finished target " + str(i+1) + " out of " + str(len(targets_cam1)))
        print("Number of iterations to get to target: " + str(count))

        if count != -1:
            counts.append(count)
        else:
            faults += 1

    filtered_counts = [x for x in counts if x != -2]
    avg_count = np.mean(filtered_counts)
    return avg_count, faults


def main():

    targets = [[740, 140]]

    # model 1
    # cams, targets -> steps
    file_name1 = "Models/isr_hit_t3_v1_model1_cam_target-steps.pth"
    X_scaler_name1 = "Models/isr_hit_t3_v1_model1_cam_target-steps_X_scaler.pkl"
    y_scaler_name1 = "Models/isr_hit_t3_v1_model1_cam_target-steps_y_scaler.pkl"
    X_scaler1 = joblib.load(X_scaler_name1)
    y_scaler1 = joblib.load(y_scaler_name1)
    model1 = FeedbackNN6(4, 2)
    model1.load_state_dict(torch.load(file_name1))
    model1.eval()


    japc.startSubscriptions()
    sleep(2)


    avg_count1, faults1 = move_to_target(model1, targets, X_scaler1, y_scaler1)
    print("finished model1")
    # move_to_references()


    japc.stopSubscriptions()
    japc.clearSubscriptions()

    print(f"avg_count1: {avg_count1:.4f}, faults1: {faults1}")

if __name__ == "__main__":
    main()
