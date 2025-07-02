import pyjapc
import pyrbac
import numpy as np
from time import sleep
import sys
import os
import pandas as pd

from Alignment.AlignMainCam import steerer
from Alignment.AlignToTarget import FeedbackNN

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from devices_lib import Camera, Motor
from AlignToTarget import BeamSteerer
import torch
from sklearn.model_selection import train_test_split


def test_steps_nn(model, X, y, input_cols, predict_target=False):
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, shuffle=True) # seed has to be the same as in training

    count = 0
    total_error = 0

    for i in range(len(X_test)):

        # current sample
        X_sample = X_test.iloc[i]
        y_sample = y_test.iloc[i]

        input_vals = np.array([X_sample[col] for col in input_cols], dtype=np.float32)
        input_tensor = torch.tensor(input_vals).unsqueeze(0)
        with torch.no_grad():
            prediction = model(input_tensor).squeeze(0).numpy()


        if predict_target:
            steerer = BeamSteerer(model, lr=0.1, iters=50)
            # todo

        # compute error
        pred = np.array(prediction)
        target = np.array(y_sample)

        error = np.linalg.norm(pred - target)
        total_error += error
        count += 1

    avg_error = total_error / count
    return avg_error

def test_target_nn(model, X, y, input_cols):
    steerer = BeamSteerer(model, lr=0.1, iters=50)


if __name__ == "__main__":
    japc = pyjapc.PyJapc(selector = '', incaAcceleratorName = '')
    client = pyrbac.AuthenticationClient()
    token = client.login_kerberos()
    japc.rbacLogin(token=token)

    M3X = Motor('TST.cfv-375-bsrdev.mot.3-4', japc, 168) #ISR
    M3Y = Motor('TST.cfv-375-bsrdev.mot.3-5', japc, 125) #ISR

    cam1 = Camera('TST.BSRH.CAM.CM', japc) #ISR

    target = [484, 304]

    df = pd.read_csv('Data/MainCamData.csv')

    # cam, m, target -> steps
    X1 = df[['CamX', 'CamY', 'Mx', 'My', 'TgtX', 'TgtY']] # has to include m
    y1 = df[['StepsX', 'StepsY']]
    input_cols1 = ['CamX', 'CamY', 'Mx', 'My', 'TgtX', 'TgtY']
    file_name1 = "Models/model1_cam_m_target-steps.pth"
    model1 = FeedbackNN(6, 2)
    model1.load_state_dict(torch.load(file_name1))
    model1.eval()
    avg_loss1 = test_steps_nn(model1, X1, y1, input_cols1)

    # cam, target -> steps
    X2 = df[['CamX', 'CamY', 'TgtX', 'TgtY', 'Mx', 'My']]
    y2 = df[['StepsX', 'StepsY']]
    input_cols2 = ['CamX', 'CamY', 'TgtX', 'TgtY']
    file_name2 = "Models/model2_cam_target-steps.pth"
    model2 = FeedbackNN(4, 2)
    model2.load_state_dict(torch.load(file_name2))
    model2.eval()
    avg_loss2 = test_steps_nn(model2, X2, y2, input_cols2)


    # # cam, steps -> target
    # X3 = df[['CamX', 'CamY', 'StepsX', 'StepsY', 'Mx', 'My']]
    # y3 = df[['TgtX', 'TgtY']]
    # input_cols3 = ['CamX', 'CamY', 'StepsX', 'StepsY']
    # file_name3 = "Models/model3_cam_steps-target.pth"
    # model3 = FeedbackNN(4, 2)
    # model3.load_state_dict(torch.load(file_name3))
    # model3.eval()
    # avg_loss3 = test_target_nn(model3, X3, y3, input_cols3, predict_target=True)


    # m, target -> steps
    X4 = df[['Mx', 'My', 'TgtX', 'TgtY']]
    y4 = df[['StepsX', 'StepsY']]
    input_cols4 = ['Mx', 'My', 'TgtX', 'TgtY']
    file_name4 = "Models/model4_m_target-steps.pth"
    model4 = FeedbackNN(4, 2)
    model4.load_state_dict(torch.load(file_name4))
    model4.eval()
    avg_loss4 = test_steps_nn(model4, X4, y4, input_cols4)


    # cam, m -> steps
    X5 = df[['CamX', 'CamY', 'Mx', 'My']]
    y5 = df[['StepsX', 'StepsY']]
    input_cols5 = ['CamX', 'CamY', 'Mx', 'My']
    file_name5 = "Models/model5_cam-steps.pth"
    model5 = FeedbackNN(4, 2)
    model5.load_state_dict(torch.load(file_name5))
    model5.eval()
    avg_loss5 = test_steps_nn(model5, X5, y5, input_cols5)


    # # cam, m, steps -> target
    # X6 = df[['CamX', 'CamY', 'Mx', 'My', 'StepsX', 'StepsY']]
    # y6 = df[['TgtX', 'TgtY']]
    # input_cols6 = ['CamX', 'CamY', 'Mx', 'My', 'StepsX', 'StepsY']
    # file_name6 = "Models/model6_cam_m_steps-target.pth"
    # model6 = FeedbackNN(6, 2)
    # model6.load_state_dict(torch.load(file_name6))
    # model6.eval()
    # avg_loss6 = test_target_nn(model6, X6, y6, input_cols6, predict_target=True)

    print("avg loss1: " + str(avg_loss1))
    print("avg loss2: " + str(avg_loss2))
    # print("avg loss3: " + str(avg_loss3))
    print("avg loss4: " + str(avg_loss4))
    print("avg loss5: " + str(avg_loss5))
    # print("avg loss6: " + str(avg_loss6))