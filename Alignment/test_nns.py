import pyjapc
import pyrbac
import numpy as np
from time import sleep
import sys
import os
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from devices_lib import Camera, Motor
from AlignToTarget import BeamSteerer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def test_steps_nn(model, X, y):
    avg_loss = 0
    return avg_loss

def test_target_nn(model, X, y):
    avg_loss = 0
    return avg_loss


if __name__ == "__main__":
    japc = pyjapc.PyJapc(selector = '', incaAcceleratorName = '')
    client = pyrbac.AuthenticationClient()
    token = client.login_kerberos()
    japc.rbacLogin(token=token)

    M3X = Motor('TST.cfv-375-bsrdev.mot.3-4', japc, 168) #ISR
    M3Y = Motor('TST.cfv-375-bsrdev.mot.3-5', japc, 125) #ISR

    cam1 = Camera('TST.BSRH.CAM.CM', japc) #ISR

    target = [484, 304]

    df = pd.read_csv('MainCamData.csv')

    # cam, m, target -> steps
    X1 = df[['CamX', 'CamY', 'Mx', 'My', 'TgtX', 'TgtY']].values
    y1 = df[['StepsX', 'StepsY']].values
    file_name1 = "model1_cam_m_target-steps.pth"
    model1 = ""
    avg_loss1 = test_steps_nn(model1, X1, y1)

    # cam, target -> steps
    X2 = df[['CamX', 'CamY', 'TgtX', 'TgtY']].values
    y2 = df[['StepsX', 'StepsY']].values
    file_name2 = "model2_cam_target-steps.pth"
    model2 = ""
    avg_loss2 = test_steps_nn(model2, X2, y2)


    # cam, steps -> target
    X3 = df[['CamX', 'CamY', 'StepsX', 'StepsY']].values
    y3 = df[['TgtX', 'TgtY']].values
    file_name3 = "model3_cam_steps-target.pth"
    model3 = ""
    avg_loss3 = test_target_nn(model3, X3, y3)


    # m, target -> steps
    X4 = df[['Mx', 'My', 'TgtX', 'TgtY']].values
    y4 = df[['StepsX', 'StepsY']].values
    file_name4 = "model4_m_target-steps.pth"
    model4 = ""
    avg_loss4 = test_steps_nn(model4, X4, y4)


    # cam, m -> steps
    X5 = df[['CamX', 'CamY', 'Mx', 'My']].values
    y5 = df[['StepsX', 'StepsY']].values
    file_name5 = "model5_cam-steps.pth"
    model5 = ""
    avg_loss5 = test_steps_nn(model5, X5, y5)

    # cam, m, steps -> target
    X6 = df[['CamX', 'CamY', 'Mx', 'My', 'StepsX', 'StepsY']].values
    y6 = df[['TgtX', 'TgtY']].values
    file_name6 = "model6_cam_m_steps-target.pth"
    model6 = ""
    avg_loss6 = test_target_nn(model6, X6, y6)

    print("avg loss1: " + str(avg_loss1))
    print("avg loss2: " + str(avg_loss2))
    print("avg loss3: " + str(avg_loss3))
    print("avg loss4: " + str(avg_loss4))
    print("avg loss5: " + str(avg_loss5))
    print("avg loss6: " + str(avg_loss6))