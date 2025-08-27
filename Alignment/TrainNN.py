## Script to perform automatic alignment of the BSRH - up to the LS as the final focus is pretty difficult to do...
#First get motor positions of everything that we will move
#Then capture the beam on all targets

########LIBRARIES############################
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
from lmfit.model import save_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from pathlib import Path

########Parameters and names#################

class FeedbackNN1(nn.Module): # dont touch main cam model uses this
    def __init__(self, input_size=8, output_size=4, seed=42):
        super(FeedbackNN1, self).__init__()
        torch.manual_seed(seed)

        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            # nn.Dropout(0.1),

            # nn.Linear(input_size, 64),
            # nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.model(x)

class FeedbackNN2(nn.Module): # do not touch this, t3 t4 uses this
    def __init__(self, input_size=8, output_size=4, seed=42):
        super(FeedbackNN2, self).__init__()
        torch.manual_seed(seed)

        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.model(x)

class FeedbackNN3(nn.Module): # do not touch this, t5 t6 uses this
    def __init__(self, input_size=4, output_size=2, seed=42):
        super(FeedbackNN3, self).__init__()
        torch.manual_seed(seed)

        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Dropout(0.1),

            # nn.Linear(64, 64),
            # nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.model(x)

class FeedbackNN4(nn.Module):  # do not touch this, t5 t6 uses this
    def __init__(self, input_size=4, output_size=2, seed=42):
        super(FeedbackNN4, self).__init__()
        torch.manual_seed(seed)
        self.model = nn.Sequential(
            nn.Linear(input_size, 1288),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(128, 128),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.model(x)

class FeedbackNN5(nn.Module):  # do not touch this, t1 t2 uses this
    def __init__(self, input_size=4, output_size=3, seed=42):
        super(FeedbackNN5, self).__init__()
        torch.manual_seed(seed)
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(128, 128),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.model(x)

class FeedbackNN6(nn.Module):  # do not touch this, t3 t4 v4 uses this
    def __init__(self, input_size=8, output_size=2, seed=42):
        super(FeedbackNN6, self).__init__()
        torch.manual_seed(seed)
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(128, 128),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.model(x)

class FeedbackNN7(nn.Module):  # do not touch this, t1 t2 uses this
    def __init__(self, input_size=4, output_size=3, seed=42):
        super(FeedbackNN7, self).__init__()
        torch.manual_seed(seed)
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(0.1),

            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.model(x)

class FeedbackNN8(nn.Module):  # do not touch this, hit t3 uses this
    def __init__(self, input_size=4, output_size=3, seed=42):
        super(FeedbackNN8, self).__init__()
        torch.manual_seed(seed)
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Dropout(0.1),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.model(x)

def train_nn(model, X, y, file_name: str, X_scaler_name: str, y_scaler_name:str, save_model=False):

    # the motor ranges might not be accurate as they differ in the opposite directions (these are the minima)
    # motor_ranges = np.array([6000, 5000, 3000])  #stepsX, stepsY, stepsPhi, stepsTheta, t1-t2
    motor_ranges = np.array([500, 650]) # t3-t4
    # motor_ranges = np.array([900, 500]) # t5-t6
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Normalize inputs and outputs
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)


    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # model = FeedbackNN(input_size=X_train.shape[1], output_size=y_train.shape[1]).to(device)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.00009, weight_decay= 1e-4)

    num_epochs = 500
    best_val_loss = float('inf')
    best_metrics = {}

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
        preds_list, targets_list = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                total_val_loss += criterion(preds, yb).item() * xb.size(0)

                preds_list.append(preds.cpu().numpy())
                targets_list.append(yb.cpu().numpy())

        val_loss = total_val_loss / len(val_loader.dataset)

        # Concatenate predictions and targets
        preds_all = np.concatenate(preds_list, axis=0)
        targets_all = np.concatenate(targets_list, axis=0)

        # Inverse normalization for RMSE calculation
        preds_real = scaler_y.inverse_transform(preds_all)
        targets_real = scaler_y.inverse_transform(targets_all)

        ## Compute RMSE global and per motor
        mse_real = mean_squared_error(targets_real, preds_real)
        rmse_global = np.sqrt(mse_real)
        per_motor_rmse = np.sqrt(np.mean((preds_real - targets_real) ** 2, axis=0))
        percent_error = per_motor_rmse / motor_ranges * 100

        print(f"Epoch {epoch:03d} — train_loss: {train_loss:.7f}, val_loss: {val_loss:.7f}")
        print(f"   RMSE Global: {rmse_global:.2f} steps")
        print(f"   Per-motor RMSE: {per_motor_rmse}")
        print(f"   % Error per motor: {percent_error}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                "rmse_global": rmse_global,
                "per_motor_rmse": per_motor_rmse,
                "percent_error": percent_error
            }
            if save_model:
                torch.save(model.state_dict(), file_name)
                joblib.dump(scaler_X, X_scaler_name)
                joblib.dump(scaler_y, y_scaler_name)

    print("\n==== Training Complete ====")
    print("Best Metrics:", best_metrics)
    return best_val_loss, best_metrics


if __name__ == "__main__":

    ################ t1 t2 V3 #####################
    # uncomment for training
    # change FeedbackNN number for different nn architecture
    # choose only one X1 based on what you want to train it on (different input) - now it's only cam2, t2

    # data_file_name = "Data/lhc_t1_t2_data_v5.csv"
    #
    # df = pd.read_csv(data_file_name)
    # print(f"There are {df.shape[0]} rows")
    #
    # # model 1
    # # cams, targets -> steps
    # X1 = df[["cam1X", "cam1Y", "cam2X", "cam2Y", "t1X", "t1Y", "t2X", "t2Y"]].values
    # # X1 = df[["cam2X", "cam2Y", "t2X", "t2Y"]].values
    # # X1 = df[["cam1X", "cam1Y", "t1X", "t1Y"]].values
    # # X1 = df[["cam1X", "cam1Y", "cam2X", "cam2Y", "mX", "mY", "mPhi", "mTheta", "t1X", "t1Y", "t2X", "t2Y"]].values
    # y1 = df[['stepsX', 'stepsPhi', 'stepsTheta']].values
    # file_name1 = "Models/lhc_t1_t2_v5_model1_cams_targets-steps.pth"
    # X_scaler_name1 = "Models/lhc_t1_t2_v5_model1_cams_targets-steps_X_scaler.pkl"
    # y_scaler_name1 = "Models/lhc_t1_t2_v5_model1_cams_targets-steps_y_scaler.pkl"
    # model1 = FeedbackNN7(X1.shape[1], y1.shape[1])
    # scaled_loss1, rmse1 = train_nn(model1, X1, y1, file_name1, X_scaler_name1, y_scaler_name1, save_model=True)


    ################ t3 t4 V4 #####################
    # uncomment for training
    # change FeedbackNN number for different nn architecture
    # choose only one X1 based on what you want to train it on (different input) - now it's only cam2, t2

    # data_file_name = "Data/isr_t3_t4_data_v4_filtered.csv"
    #
    # df = pd.read_csv(data_file_name)
    # print(f"There are {df.shape[0]} rows")
    #
    # # model 1
    # # cams, targets -> steps
    # # X1 = df[["cam1X", "cam1Y", "cam2X", "cam2Y", "t1X", "t1Y", "t2X", "t2Y"]].values
    # X1 = df[["cam2X", "cam2Y", "t2X", "t2Y"]].values
    # # X1 = df[["cam1X", "cam1Y", "t1X", "t1Y"]].values
    # y1 = df[['stepsPhi', 'stepsTheta']].values
    # file_name1 = "Models/isr_t3_t4_v4_filtered_model3_only_cam2_target2-steps.pth"
    # X_scaler_name1 = "Models/isr_t3_t4_v4_filtered_model3_only_cam2_target2-steps_X_scaler.pkl"
    # y_scaler_name1 = "Models/isr_t3_t4_v4_filtered_model3_only_cam2_target2-steps_y_scaler.pkl"
    # model1 = FeedbackNN6(X1.shape[1], y1.shape[1])
    # scaled_loss1, rmse1 = train_nn(model1, X1, y1, file_name1, X_scaler_name1, y_scaler_name1, save_model=True)


    ###################### t5 t6 V3 ######################
    # uncomment for training
    # change FeedbackNN number for different nn architecture

    # data_file_name = "Data/isr_t5_t6_data_v3.csv"

    # df = pd.read_csv(data_file_name)
    # print(f"There are {df.shape[0]} rows")

    # model 1
    # cams, targets -> steps
    # X1 = df[["cam2X", "cam2Y", "t2X", "t2Y"]].values
    # y1 = df[['stepsPhi', 'stepsTheta']].values
    # file_name1 = "Models/isr_t5_t6_v3_model1_cam2_target2-steps.pth"
    # X_scaler_name1 = "Models/isr_t5_t6_v3_model1_cam2_target2-steps_X_scaler.pkl"
    # y_scaler_name1 = "Models/isr_t5_t6_v3_model1_cam2_target2-steps_y_scaler.pkl"
    # model1 = FeedbackNN6(X1.shape[1], y1.shape[1])
    # scaled_loss1, rmse1 = train_nn(model1, X1, y1, file_name1, X_scaler_name1, y_scaler_name1, save_model=True)


    ###################### hit t3 ######################
    # uncomment for training
    # change FeedbackNN number for different nn architecture

    data_file_name = "Data/isr_t3_t4_data_v4_filtered_scaled.csv"

    df = pd.read_csv(data_file_name)
    print(f"There are {df.shape[0]} rows")

    # model 1
    # cams, targets -> steps
    X1 = df[["cam1X", "cam1Y", "t1X", "t1Y"]].values
    y1 = df[['stepsPhi', 'stepsTheta']].values
    file_name1 = "Models/isr_hit_t3_model1_cam_target-steps.pth"
    X_scaler_name1 = "Models/isr_hit_t3_model1_cam_target-steps_X_scaler.pkl"
    y_scaler_name1 = "Models/isr_hit_t3_model1_cam_target-steps_y_scaler.pkl"
    model1 = FeedbackNN6(X1.shape[1], y1.shape[1])
    scaled_loss1, rmse1 = train_nn(model1, X1, y1, file_name1, X_scaler_name1, y_scaler_name1, save_model=False)