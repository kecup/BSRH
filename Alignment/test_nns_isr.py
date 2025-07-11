import pyjapc
import sys
import os
import pyrbac
import numpy as np
import torch
from time import sleep

from pyarrow.lib import float64
from triton.language import dtype

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from devices_lib import Camera, Motor
from AlignToTarget import Aligner, BeamSteerer, FeedbackNN


def generate_targets():
    # targets = [[484, 304], [500, 350], [109, 512], [168, 136], [775, 123], [890, 496]] # just for testing now
    # targets = [[67, 532], [74, 332], [67, 81], [473, 74], [863, 75], [888, 293], [909, 551], [457, 535], [484, 304]] # too on edge
    targets = [[145, 466], [135, 135], [800, 172], [814, 480], [106, 290], [477, 113], [801, 313], [470, 483],
               [282, 434], [775, 205], [697, 420], [308, 238], [484, 304]]
    return targets

def steer_to_target(model, targets, motor_x, motor_y, camera, ref_x, ref_y, dist_tolerance=14):
    counts = []
    faults = 0
    steerer = BeamSteerer(model, lr=0.1, iters=50)
    for i in range(len(targets)):
        target = targets[i]
        aligner = Aligner(motor_x, motor_y, camera, target, model, steerer, ref_x, ref_y)
        count = aligner.align_testing(dist_tolerance)
        print("finished target " + str(i+1) + " out of " + str(len(targets)))
        print("Number of iterations to get to target: " + str(count))
        if count != -1 and count != -2:
            counts.append(count)
        elif count == -2:
            faults += 1

    filtered_counts = [x for x in counts if x != -2]
    avg_count = np.mean(filtered_counts)

    return avg_count, faults

def move_to_target(model, targets, motor_x, motor_y, camera, ref_x, ref_y, dist_tolerance=14):
    counts = []
    faults = 0
    for i in range(len(targets)):
        target = np.array(targets[i], dtype=np.float32)
        count = 0
        cam_pos = np.array(camera.getBeamPosition(), dtype=np.float32)
        while np.abs(cam_pos[0] - target[0]) > dist_tolerance and np.abs(cam_pos[1] - target[1]) > dist_tolerance:
            if count > 30:
                print("Not getting closer reasonably fast.")
                faults += 1
                count = -2
                break
            if (cam_pos[0] == -1 or cam_pos[1] == -1):
                print("beam lost")
                print("Sending to references.")
                count = -1
                motor_x.setAbsPosition(ref_x)
                motor_y.setAbsPosition(ref_y)
                busy_counter = 0
                while motor_x.isBusy() or motor_y.isBusy():
                    sleep(0.2)
                    busy_counter += 1
                    if busy_counter > 40:
                        break
                break
            input_vector = np.concatenate((cam_pos, target), axis=0, dtype=np.float32())
            input_tensor = torch.tensor(input_vector).unsqueeze(0)
            with torch.no_grad():
                prediction = model(input_tensor).squeeze(0).numpy() # steps
            motor_x.setRelPosition(int(prediction[0])) # move to prediction
            motor_y.setRelPosition(int(prediction[1])) # move to prediction
            busy_counter = 0
            while motor_x.isBusy() or motor_y.isBusy():
                sleep(0.2)
                busy_counter += 1
                if busy_counter > 40:
                    break
            cam_pos = np.array(camera.getBeamPosition(), dtype=np.float32) # new camera position
            count += 1
        print("finished target " + str(i+1) + " out of " + str(len(targets)))
        print("Number of iterations to get to target: " + str(count))

        if count != -1:
            counts.append(count)
        else:
            faults += 1

    filtered_counts = [x for x in counts if x != -2]
    avg_count = np.mean(filtered_counts)
    return avg_count, faults

def move_motors_to_start(motor_x, motor_y, start_pos):
    # move motors to starting position
    motor_x.setAbsPosition(start_pos[0])
    motor_y.setAbsPosition(start_pos[1])
    busy_counter = 0
    while motor_x.isBusy() or motor_y.isBusy():
        sleep(0.2)
        busy_counter += 1
        if busy_counter > 70:
            break

def main():
    japc = pyjapc.PyJapc(selector = '', incaAcceleratorName = '')
    client = pyrbac.AuthenticationClient()
    token = client.login_kerberos()
    japc.rbacLogin(token=token)

    M3X = Motor('TST.cfv-375-bsrdev.mot.3-4', japc, 168) #ISR
    M3Y = Motor('TST.cfv-375-bsrdev.mot.3-5', japc, 125) #ISR

    cam1 = Camera('TST.BSRH.CAM.CM', japc) #ISR

    M3X.setRelPosition(168)
    M3Y.setRelPosition(125)

    # model 1
    # cam, steps -> target
    file_name1 = "Models/model3_cam_steps-target.pth"
    model1 = FeedbackNN(4, 2)
    model1.load_state_dict(torch.load(file_name1))
    model1.eval()


    # model 2
    # cam, m, steps -> target
    file_name2 = "Models/model6_cam_m_steps-target.pth"
    model2 = FeedbackNN(6, 2)
    model2.load_state_dict(torch.load(file_name2))
    model2.eval()

    # model 3
    # original model
    file_name3 = "Models/CM_feedbackModel.pth"
    model3 = FeedbackNN(4, 2)
    model3.load_state_dict(torch.load(file_name3))
    model3.eval()

    # model 4
    # cam, target -> steps
    file_name4 = "Models/model2_cam_target-steps.pth"
    model4 = FeedbackNN(4, 2)
    model4.load_state_dict(torch.load(file_name4))
    model4.eval()

    targets = generate_targets()
    start_pos = [140600, 163500]

    japc.startSubscriptions()

    move_motors_to_start(M3X, M3Y, start_pos)
    avg_count1, faults1 = steer_to_target(model1, targets=targets, motor_x=M3X, motor_y=M3Y, camera=cam1,
                                          ref_x=start_pos[0], ref_y=start_pos[1], dist_tolerance=10)
    print("finished model1")
    # move_motors_to_start(M3X, M3Y, start_pos)
    # avg_count2, faults2 = steer_to_target(model2, targets=targets, motor_x=M3X, motor_y=M3Y, camera=cam1, dist_tolerance=14)
    # print("finished model2")
    # move_motors_to_start(M3X, M3Y, start_pos)
    # avg_count3, faults3 = steer_to_target(model3, targets=targets, motor_x=M3X, motor_y=M3Y, camera=cam1, dist_tolerance=14)
    # print("finished model3")
    move_motors_to_start(M3X, M3Y, start_pos)
    avg_count4, faults4 = move_to_target(model4, targets=targets, motor_x=M3X, motor_y=M3Y, camera=cam1,
                                          ref_x=start_pos[0], ref_y=start_pos[1], dist_tolerance=10)
    print("finished model4")
    move_motors_to_start(M3X, M3Y, start_pos)

    japc.stopSubscriptions()
    japc.clearSubscriptions()

    print(f"avg_count1: {avg_count1:.4f}, faults1: {faults1}")
    # print(f"avg_count2: {avg_count2:.4f}, faults2: {faults2}") # current mx, my too different from training data, doesn't work well
    # print(f"avg_count3: {avg_count3:.4f}, faults3: {faults3}")
    print(f"avg_count4: {avg_count4:.4f}, faults4: {faults4}")

if __name__ == "__main__":
    main()
