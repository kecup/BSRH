import pyjapc
import sys
import os
import pyrbac
import numpy as np
import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from devices_lib import Camera, Motor
from AlignToTarget import Aligner, BeamSteerer, FeedbackNN


def generate_targets():
    targets = [[484, 304], [500, 350]] # just for testing now
    return targets

def steer_to_target(model, targets, motor_x, motor_y, camera, dist_tolerance=14):
    counts = []
    faults = 0
    steerer = BeamSteerer(model, lr=0.1, iters=30)
    for i in range(len(targets)):
        target = targets[i]
        aligner = Aligner(motor_x, motor_y, camera, target, model, steerer)
        count = aligner.align_testing(dist_tolerance)
        if count != -1 and count != -2:
            counts.append(count)
        elif count == -2:
            faults += 1
    avg_count = np.mean(counts)

    return avg_count, faults

def main():
    japc = pyjapc.PyJapc(selector = '', incaAcceleratorName = '')
    client = pyrbac.AuthenticationClient()
    token = client.login_kerberos()
    japc.rbacLogin(token=token)

    M3X = Motor('TST.cfv-375-bsrdev.mot.3-4', japc, 168) #ISR
    M3Y = Motor('TST.cfv-375-bsrdev.mot.3-5', japc, 125) #ISR

    cam1 = Camera('TST.BSRH.CAM.CM', japc) #ISR

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

    targets = generate_targets()

    japc.startSubscriptions()

    avg_count1, faults1 = steer_to_target(model1, targets=targets, motor_x=M3X, motor_y=M3Y, camera=cam1, dist_tolerance=14)
    # avg_count2, faults2 = steer_to_target(model2, targets=targets, motor_x=M3X, motor_y=M3Y, camera=cam1, dist_tolerance=14)

    japc.stopSubscriptions()
    japc.clearSubscriptions()

    print(f"avg_count1: {avg_count1:.4f}, faults1: {faults1}")
    # print(f"avg_count2: {avg_count2:.4f}, faults2: {faults2}")

if __name__ == "__main__":
    main()
