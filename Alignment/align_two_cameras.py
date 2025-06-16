import pyjapc
import pyrbac
from time import sleep
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import select
from devices_lib import Camera, Motor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


japc = pyjapc.PyJapc(selector = '', incaAcceleratorName = '')
client = pyrbac.AuthenticationClient()
token = client.login_kerberos()
japc.rbacLogin(token=token)


class FeedbackNN(nn.Module):
    def __init__(self, state_size=8, action_size=4, seed=45):
        super(FeedbackNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class BeamSteerer:
    def __init__(self, forward_model: nn.Module, lr=0.1, iters=30, lambda_reg=0.01):
        self.f_phi = forward_model
        self.lr = lr
        self.iters = iters
        self.lambda_reg = lambda_reg  # penalty strength for motor movement

    def compute_control(self, x0: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        x0: tensor([cam1_x, cam1_y, cam2_x, cam2_y, motorX_pos, motorY_pos, phi_pos, theta_pos])
        target: tensor([cam1_target_x, cam1_target_y, cam2_target_x, cam2_target_y])
        returns: optimized u (tensor([Δx, Δy, Δphi, Δtheta]))
        """
        u = torch.zeros(4, requires_grad=True)  # control for 4 motors

        opt = optim.SGD([u], lr=self.lr)

        for _ in range(self.iters):
            opt.zero_grad()
            xu = torch.cat([x0, u], dim=0).unsqueeze(0)  # input to model
            x_pred = self.f_phi(xu).squeeze(0)  # predicted beam positions on cameras

            mse_loss = nn.functional.mse_loss(x_pred, target)  # beam error
            reg_loss = self.lambda_reg * torch.sum(u ** 2)  # step penalty

            loss = mse_loss + reg_loss
            loss.backward()
            opt.step()

        return u.detach()

    def step(self, motors, cameras, targets):
        # Read current camera beam positions
        cam1_pos = cameras[0].getBeamPosition()  # [x, y]
        cam2_pos = cameras[1].getBeamPosition()  # [x, y]

        # Current motor positions
        motor_positions = [m.getPosition() for m in motors]  # [X, Y, Phi, Theta]

        # Build state tensor
        x0 = torch.tensor(cam1_pos + cam2_pos + motor_positions, dtype=torch.float32)

        # Build target tensor
        target = torch.tensor(targets[0] + targets[1], dtype=torch.float32)  # 4 values

        # Compute control (motor step changes)
        u_opt = self.compute_control(x0, target)  # ΔX, ΔY, ΔPhi, ΔTheta

        # Send movements to motors
        for i, motor in enumerate(motors):
            motor.setRelPosition(u_opt[i].item())  # convert tensor to float


# Define motors
# M_X = Motor('LHC.BSRH.M3-PHI.B2', backlash_steps=35)
# M_Y = Motor('LHC.BSRH.M3-THETA.B2', backlash_steps=30)
# M_Phi = Motor('LHC.BSRH.LENS-PHI.B2', backlash_steps=20)
# M_Theta = Motor('LHC.BSRH.LENS-THETA.B2', backlash_steps=20)

M_X = Motor('TST.cfv-375-bsrdev.mot.3-2', japc, backlash_steps=5)
M_Y = Motor('TST.cfv-375-bsrdev.mot.3-3', japc, backlash_steps=5)
M_Phi = Motor('TST.cfv-375-bsrdev.mot.3-4', japc, backlash_steps=168)
M_Theta = Motor('TST.cfv-375-bsrdev.mot.3-5', japc, backlash_steps=125)

motors = [M_X, M_Y, M_Phi, M_Theta]

# Define cameras
# cam1 = Camera('LHC.BSRH.CAM.CM1.B2')
# cam2 = Camera('LHC.BSRH.CAM.CM2.B2')

cam1 = Camera('TST.BSRH.CAM.CM', japc)
cam2 = Camera('TST.BSRH.MON.M2', japc)

cameras = [cam1, cam2]

# Targets for each camera
target1 = [484, 304]  # cam1 target (x, y)
target2 = [550, 210]  # cam2 target (x, y)

targets = [target1, target2]

# Load trained model
model = FeedbackNN()
model.load_state_dict(torch.load('two_camera_model.pth'))
model.eval()

steerer = BeamSteerer(model)

japc.startSubscriptions()

done = False
while True:
    if select.select([sys.stdin], [], [], 0.0)[0]:
        print("Key pressed! Exiting loop.")
        break

    # Compute optimal motor movements
    steerer.step(motors, cameras, targets)
    print("Step executed.")
    sleep(2)

japc.stopSubscriptions()