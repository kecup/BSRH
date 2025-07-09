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

########Parameters and names#################

class FeedbackNN(nn.Module): #Not used here
  def __init__(self, state_size, action_size, seed = 45) -> None:
      self.NNsize = 16
      super(FeedbackNN, self).__init__()
      self.seed = torch.manual_seed(seed)
      self.fc1 = nn.Linear(state_size, self.NNsize) #First Neuron layer
      self.fc2 = nn.Linear(self.NNsize,self.NNsize)#Next Neuron layer
      self.fc3 = nn.Linear(self.NNsize,self.NNsize)#Next Neuron layer
      self.fc4 = nn.Linear(self.NNsize, action_size)#Last Neuron layer

  def forward(self, state):
      x = self.fc1(state) #output of the first layer
      x = F.relu(x) # rectify function of the output

      y = self.fc2(x)
      y = F.relu(y)

      z = self.fc4(y)
      return z


def train_nn(X, y, file_name: str):

    data_train, data_val, delta_train, delta_val = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle = True)

    ##Scaler to normalize?
    scaler = StandardScaler()

    #data_train = scaler.fit_transform(data_train)
    #data_val = scaler.transform(data_val)

    train_ds = TensorDataset(torch.from_numpy(data_train).float(), torch.from_numpy(delta_train).float())
    val_ds = TensorDataset(torch.from_numpy(data_val).float(), torch.from_numpy(delta_val).float())

    train_loader = DataLoader(train_ds, batch_size=32, shuffle = True)
    val_loader   = DataLoader(val_ds,   batch_size=32)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FeedbackNN(data_train.shape[1],2).to(device)
    #model = ResidualMLP(6,2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)#, weight_decay= 1e-4)

    num_epochs = 100
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
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                total_val_loss += criterion(preds, yb).item() * xb.size(0)
        val_loss = total_val_loss / len(val_loader.dataset)

        print(f"Epoch {epoch:03d} — train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

        # — Optional: save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), file_name)
        #    joblib.dump(scaler, 'scaler.pkl')

    return best_val_loss


if __name__ == "__main__":

    ##Open a csv file that we will use to save the training data:
    df = pd.read_csv('Data/BackLashData.csv')

    # bad
    # cam, m, target -> steps
    X1 = df[['CamX', 'CamY', 'MotX', 'MotY', 'TgtX', 'TgtY']].values
    y1 = df[['StepsX', 'StepsY']].values
    file_name1 = "Models/model1_cam_m_target-steps.pth"
    loss1 = train_nn(X1,y1, file_name1)
    print("Finished 1st model.")
    print("loss: " + str(loss1))

    # bad
    # cam, target -> steps
    X2 = df[['CamX', 'CamY', 'TgtX', 'TgtY']].values
    y2 = df[['StepsX', 'StepsY']].values
    file_name2 = "Models/model2_cam_target-steps.pth"
    loss2 = train_nn(X2, y2, file_name2)
    print("Finished 2nd model.")
    print("loss: " + str(loss2))

    # current working version
    # cam, steps -> target
    X3 = df[['CamX', 'CamY', 'StepsX', 'StepsY']].values
    y3 = df[['TgtX', 'TgtY']].values
    file_name3 = "Models/model3_cam_steps-target.pth"
    loss3 = train_nn(X3, y3, file_name3)
    print("Finished 3rd model.")
    print("loss: " + str(loss3))

    # bad
    # m, target -> steps
    X4 = df[['MotX', 'MotY', 'TgtX', 'TgtY']].values
    y4 = df[['StepsX', 'StepsY']].values
    file_name4 = "Models/model4_m_target-steps.pth"
    loss4 = train_nn(X4, y4, file_name4)
    print("Finished 4th model.")
    print("loss: " + str(loss4))

    # # doesn't make much sense
    # # cam, m -> steps
    # X5 = df[['CamX', 'CamY', 'Mx', 'My']].values
    # y5 = df[['StepsX', 'StepsY']].values
    # file_name5 = "model5_cam-steps.pth"
    # loss5 = train_nn(X5, y5, file_name5)
    # print("Finished 5th model.")
    # print("loss: " + str(loss5))

    # mx, my changing compared to train data
    # cam, m, steps -> target
    X6 = df[['CamX', 'CamY', 'MotX', 'MotY', 'StepsX', 'StepsY']].values
    y6 = df[['TgtX', 'TgtY']].values
    file_name6 = "Models/model6_cam_m_steps-target.pth"
    loss6 = train_nn(X6, y6, file_name6)
    print("Finished 6th model.")
    print("loss: " + str(loss6))