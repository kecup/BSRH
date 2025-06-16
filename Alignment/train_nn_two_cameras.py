import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import joblib

######## Parameters and names #################

class FeedbackNN(nn.Module):
    def __init__(self, state_size=8, action_size=4, seed=45) -> None:
        super(FeedbackNN, self).__init__()
        self.NNsize = 32  # increased neurons to handle more input complexity
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, self.NNsize)
        self.fc2 = nn.Linear(self.NNsize, self.NNsize)
        self.fc3 = nn.Linear(self.NNsize, self.NNsize)
        self.fc4 = nn.Linear(self.NNsize, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# Load CSV data (adjust column names accordingly)
df = pd.read_csv('two_cam_data.csv') # to be created

# Expected columns: Cam1X, Cam1Y, Cam2X, Cam2Y, StepsX, StepsY, StepsPhi, StepsTheta, Tgt1X, Tgt1Y, Tgt2X, Tgt2Y
input_cols = ['Cam1X', 'Cam1Y', 'Cam2X', 'Cam2Y', 'StepsX', 'StepsY', 'StepsPhi', 'StepsTheta']
output_cols = ['Tgt1X', 'Tgt1Y', 'Tgt2X', 'Tgt2Y']

data = df[input_cols].values
delta = df[output_cols].values

# Split dataset
data_train, data_val, delta_train, delta_val = train_test_split(
    data, delta, test_size=0.2, random_state=42, shuffle=True)

scaler = StandardScaler()
data_train = scaler.fit_transform(data_train)
data_val = scaler.transform(data_val)
joblib.dump(scaler, 'scaler.pkl')

train_ds = TensorDataset(torch.from_numpy(data_train).float(),
                         torch.from_numpy(delta_train).float())
val_ds = TensorDataset(torch.from_numpy(data_val).float(),
                       torch.from_numpy(delta_val).float())

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FeedbackNN(state_size=8, action_size=4).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 100
best_val_loss = float('inf')

for epoch in range(1, num_epochs + 1):
    # --- TRAIN ---
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

    # --- VALIDATE ---
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            total_val_loss += criterion(preds, yb).item() * xb.size(0)
    val_loss = total_val_loss / len(val_loader.dataset)

    print(f"Epoch {epoch:03d} — train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'two_camera_model.pth')
        # joblib.dump(scaler, 'scaler.pkl')  # If scaling enabled