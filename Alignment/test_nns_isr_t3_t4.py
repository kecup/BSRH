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
from TrainNN import FeedbackNN1, FeedbackNN2

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
CAM1_TARGET = [751, 144] # cam1 target centre

def generate_targets():
    # targets = [[484, 304], [500, 350], [109, 512], [168, 136], [775, 123], [890, 496]] # just for testing now
    # targets = [[67, 532], [74, 332], [67, 81], [473, 74], [863, 75], [888, 293], [909, 551], [457, 535], [484, 304]] # too on edge
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

def move_to_target(model, targets_cam2, X_scaler, y_scaler, dist_tolerance=14):
    counts = []
    faults = 0
    for i in range(len(targets_cam2)):
        target = np.array(targets_cam2[i], dtype=np.float32)
        count = 0
        cam1_pos = np.array(cam1.getBeamPosition(), dtype=np.float32)
        cam2_pos = np.array(cam2.getBeamPosition(), dtype=np.float32)
        while (np.abs(cam2_pos[0] - target[0]) > dist_tolerance and np.abs(cam2_pos[1] - target[1]) > dist_tolerance and
               np.abs(cam1_pos[0] - CAM1_TARGET[0]) > dist_tolerance and np.abs(cam1_pos[1] - CAM1_TARGET[1]) > dist_tolerance):
            if count > 30:
                print("Not getting closer reasonably fast.")
                faults += 1
                count = -2
                break
            if (cam2_pos[0] == -1 or cam2_pos[1] == -1):
                print("beam lost")
                print("Sending to references.")
                count = -1
                break

            input_vector = np.concatenate((cam1_pos, cam2_pos, CAM1_TARGET, target), axis=0, dtype=np.float32())
            scaled_input = X_scaler.transform(input_vector)
            input_tensor = torch.tensor(input_vector).unsqueeze(0)
            with torch.no_grad():
                # inp_tensor = torch.from_numpy(scaled_input).float()
                prediction = model(input_tensor).squeeze(0).numpy() # steps

            prediction_real = y_scaler.inverse_transform(prediction)
            move_motors(prediction_real[0], prediction_real[1], prediction_real[2], prediction_real[3])

            cam1_pos = np.array(cam1.getBeamPosition(), dtype=np.float32) # new camera position
            cam2_pos = np.array(cam2.getBeamPosition(), dtype=np.float32) # new camera position
            count += 1
        print("finished target " + str(i+1) + " out of " + str(len(targets_cam2)))
        print("Number of iterations to get to target: " + str(count))

        if count != -1:
            counts.append(count)
        else:
            faults += 1

    filtered_counts = [x for x in counts if x != -2]
    avg_count = np.mean(filtered_counts)
    return avg_count, faults

def main():

    targets_cam2 = generate_targets()

    # model 1
    # cams, targets -> steps
    file_name1 = "Models/isr_t3_t4_v2(copy17254)_filtered_model1_cams_targets-steps.pth"
    X_scaler_name1 = "Models/isr_t3_t4_v2(copy17254)_filtered_model1_cams_targets-steps_X_scaler.pkl"
    y_scaler_name1 = "Models/isr_t3_t4_v2(copy17254)_filtered_model1_cams_targets-steps_X_scaler.pkl"
    X_scaler1 = joblib.load(X_scaler_name1)
    y_scaler1 = joblib.load(y_scaler_name1)
    model1 = FeedbackNN1(8, 4)
    model1.load_state_dict(torch.load(file_name1))
    model1.eval()

    # model 2
    # cams, targets -> steps
    file_name2 = "Models/isr_t3_t4_v2(copy17254)_filtered_model2_cams_targets-steps.pth"
    X_scaler_name2 = "Models/isr_t3_t4_v2(copy17254)_filtered_model2_cams_targets-steps_X_scaler.pkl"
    y_scaler_name2 = "Models/isr_t3_t4_v2(copy17254)_filtered_model2_cams_targets-steps_X_scaler.pkl"
    X_scaler2 = joblib.load(X_scaler_name2)
    y_scaler2 = joblib.load(y_scaler_name2)
    model2 = FeedbackNN2(8, 4)
    model2.load_state_dict(torch.load(file_name2))
    model2.eval()

    japc.startSubscriptions()
    sleep(2)

    move_to_references()
    avg_count1, faults1 = move_to_target(model1, targets_cam2, X_scaler1, y_scaler1, dist_tolerance=10)
    print("finished model1")
    move_to_references()
    avg_count2, faults2 = move_to_target(model2, targets_cam2, X_scaler2, y_scaler2, dist_tolerance=10)
    print("finished model2")
    move_to_references()


    japc.stopSubscriptions()
    japc.clearSubscriptions()

    print(f"avg_count1: {avg_count1:.4f}, faults1: {faults1}")
    print(f"avg_count2: {avg_count2:.4f}, faults2: {faults2}")


if __name__ == "__main__":
    main()



