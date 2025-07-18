## Script to perform automatic alignment of the BSRH - up to the LS as the final focus is pretty difficult to do...
#First get motor positions of everything that we will move
#Then capture the beam on all targets

########LIBRARIES############################
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from pathlib import Path

########Parameters and names#################

class FeedbackNN(nn.Module): #Not used here
  def __init__(self, state_size, action_size, seed = 45) -> None:
      self.NNsize = 16
      super(FeedbackNN, self).__init__()
      self.seed = torch.manual_seed(seed)
      self.dropout = nn.Dropout(p=0.1)
      self.fc1 = nn.Linear(state_size, self.NNsize) #First Neuron layer
      self.fc2 = nn.Linear(self.NNsize,self.NNsize)#Next Neuron layer
      self.fc3 = nn.Linear(self.NNsize,self.NNsize)#Next Neuron layer
      self.fc4 = nn.Linear(self.NNsize, action_size)#Last Neuron layer

def forward(self, state):
    x = self.fc1(state) #output of the first layer
    x = F.relu(x) # rectify function of the output
    #   x = self.dropout(x)

    y = self.fc2(x)
    y = F.relu(y)
    #   y = self.dropout(y)

    z = self.fc4(y)
    return z


class FeedbackNN2(nn.Module):
    def __init__(self, input_size=8, output_size=4, seed=42):
        super(FeedbackNN2, self).__init__()
        torch.manual_seed(seed)
        self.dropout = nn.Dropout(p=0.1)

        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        # self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(16, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # x = F.relu(self.fc3(x))
        return self.fc4(x)


def train_nn(X, y, file_name: str):

    data_train, data_val, delta_train, delta_val = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle = True)

    ##Scaler to normalize?
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    data_train = scaler_X.fit_transform(data_train)
    data_val = scaler_X.transform(data_val)

    delta_train = scaler_y.fit_transform(delta_train)
    delta_val = scaler_y.transform(delta_val)

    train_ds = TensorDataset(torch.from_numpy(data_train).float(), torch.from_numpy(delta_train).float())
    val_ds = TensorDataset(torch.from_numpy(data_val).float(), torch.from_numpy(delta_val).float())

    train_loader = DataLoader(train_ds, batch_size=32, shuffle = True)
    val_loader   = DataLoader(val_ds,   batch_size=32)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FeedbackNN2(data_train.shape[1],delta_train.shape[1]).to(device)
    #model = ResidualMLP(6,2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay= 1e-4)

    num_epochs = 200
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs+1):
        # ——— TRAIN ———
        print("Epoch: " + str(epoch))
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        # ——— VALIDATE ———
        model.eval()
        total_val_loss = 0
        preds_list = []
        targets_list = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                total_val_loss += criterion(preds, yb).item() * xb.size(0)

                preds_list.append(preds.cpu().numpy())
                targets_list.append(yb.cpu().numpy())

        val_loss = total_val_loss / len(val_loader.dataset)

        # ——— Convert and inverse-transform to real units ———
        preds_all = np.concatenate(preds_list, axis=0)
        targets_all = np.concatenate(targets_list, axis=0)

        preds_real = scaler_y.inverse_transform(preds_all)
        targets_real = scaler_y.inverse_transform(targets_all)

        # ——— Compute RMSE in real units ———
        mse_real = mean_squared_error(targets_real, preds_real)
        rmse_real = np.sqrt(mse_real)
        print(
            f"Epoch {epoch:03d} — train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, RMSE: {rmse_real:.2f}, MSE: {mse_real:.2f}")


        # — Optional: save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_mse = mse_real
            torch.save(model.state_dict(), file_name)
            joblib.dump(scaler_X, script_dir / "Models" / "isr_model2_t3_t4_X_scaler.pkl")
            joblib.dump(scaler_y, script_dir / "Models" / "isr_model2_t3_t4_y_scaler.pkl")

    return best_val_loss, best_mse


if __name__ == "__main__":

    ##Open a csv file that we will use to save the training data:
    # df = pd.read_csv('Data/BackLashData.csv')
    # df = pd.read_csv('Data/MainCamData.csv')

    # # cam, m, target -> steps
    # X1 = df[['CamX', 'CamY', 'MotX', 'MotY', 'TgtX', 'TgtY']].values
    # y1 = df[['StepsX', 'StepsY']].values
    # file_name1 = "Models/model1_cam_m_target-steps.pth"
    # loss1 = train_nn(X1,y1, file_name1)
    # print("Finished 1st model.")
    # print("loss: " + str(loss1))

    # bad
    # cam, target -> steps
    # X2 = df[['CamX', 'CamY', 'TgtX', 'TgtY']].values
    # y2 = df[['StepsX', 'StepsY']].values
    # file_name2 = "Models/model2_cam_target-steps.pth"
    # loss2 = train_nn(X2, y2, file_name2)
    # print("Finished 2nd model.")
    # print("loss: " + str(loss2))

    # current working version
    # cam, steps -> target
    # X3 = df[['CamX', 'CamY', 'StepsX', 'StepsY']].values
    # y3 = df[['TgtX', 'TgtY']].values
    # file_name3 = "Models/model3_cam_steps-target.pth"
    # loss3 = train_nn(X3, y3, file_name3)
    # print("Finished 3rd model.")
    # print("loss: " + str(loss3))
    #
    # # bad
    # # m, target -> steps
    # X4 = df[['MotX', 'MotY', 'TgtX', 'TgtY']].values
    # y4 = df[['StepsX', 'StepsY']].values
    # file_name4 = "Models/model4_m_target-steps.pth"
    # loss4 = train_nn(X4, y4, file_name4)
    # print("Finished 4th model.")
    # print("loss: " + str(loss4))
    #
    # # # doesn't make much sense
    # # # cam, m -> steps
    # # X5 = df[['CamX', 'CamY', 'Mx', 'My']].values
    # # y5 = df[['StepsX', 'StepsY']].values
    # # file_name5 = "model5_cam-steps.pth"
    # # loss5 = train_nn(X5, y5, file_name5)
    # # print("Finished 5th model.")
    # # print("loss: " + str(loss5))
    #
    # # mx, my changing compared to train data
    # # cam, m, steps -> target
    # X6 = df[['CamX', 'CamY', 'MotX', 'MotY', 'StepsX', 'StepsY']].values
    # y6 = df[['TgtX', 'TgtY']].values
    # file_name6 = "Models/model6_cam_m_steps-target.pth"
    # loss6 = train_nn(X6, y6, file_name6)
    # print("Finished 6th model.")
    # print("loss: " + str(loss6))

    script_dir = Path(__file__).parent
    data_path = script_dir / "Data" / "isr_t3_t4_data.csv"
    # data_file_name = "Data/isr_t3_t4_data.csv"

    df2 = pd.read_csv(data_path) # change slash windows/linux
    df2 = df2[~df2[['cam2X', 'cam2Y']].isin([-1]).any(axis=1)]



    # # cam1, cam2, target1, target2 -> steps
    # X1 = df2[['cam1X', 'cam1Y', 'cam2X', 'cam2Y', 't1X', 't1Y', 't2X', 't2Y']].values
    # y1 = df2[['stepsX', 'stepsY', 'stepsPhi', 'stepsTheta']].values
    # file_name1 = "Models/isr_t3_t4_model1_cam_target-steps.pth"
    # scaled_loss1, mse1 = train_nn(X1, y1, file_name1)
    # print("Finished training model1.")
    # print(f"Best scaled loss: {scaled_loss1:.2f}, MSE loss: {mse1:.2f}")

    # its is made for windows now
    # cam2, target2 -> steps
    X2 = df2[['cam2X', 'cam2Y','t2X', 't2Y']].values
    y2 = df2[['stepsX', 'stepsY', 'stepsPhi', 'stepsTheta']].values
    file_name2 = script_dir / "Models"  / "isr_t3_t4_model2_only_cam2_target-steps.pth"
    scaled_loss2, mse2 = train_nn(X2, y2, file_name2)
    print("Finished training model2.")
    print(f"Best scaled loss: {scaled_loss2:.2f}, MSE loss: {mse2:.2f}")