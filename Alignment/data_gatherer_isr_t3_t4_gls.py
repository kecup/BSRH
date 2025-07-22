# gather data on isr - mirror2 (x, y, phi, theta) + target3 (see-through) + target4
# use guided local search to cover the region of cam1

import pyjapc
import csv
import random
from time import sleep
import sys
import os
import numpy as np
import itertools

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from devices_lib import Camera, Motor, Pneumatics
import pyrbac

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

# Step sizes
STEP_SIZE = {
    'x': 50,
    'y': 50,
    'phi': 50,
    'theta': 50
}

# Cam1 bounding box
CAM1_BOUNDS = {
    "x_min": 659,
    "x_max": 844,
    "y_min": 51,
    "y_max": 237
}


# Define all small moves
MOVES = [
    (+STEP_SIZE['x'], 0, 0, 0), (-STEP_SIZE['x'], 0, 0, 0),
    (0, +STEP_SIZE['y'], 0, 0), (0, -STEP_SIZE['y'], 0, 0),
    (0, 0, +STEP_SIZE['phi'], 0), (0, 0, -STEP_SIZE['phi'], 0),
    (0, 0, 0, +STEP_SIZE['theta']), (0, 0, 0, -STEP_SIZE['theta']),
]

def move_motors(stepsX: int, stepsY: int, stepsPhi: int, stepsTheta: int):
    motor_x.setRelPosition(stepsX)
    motor_y.setRelPosition(stepsY)
    motor_phi.setRelPosition(stepsPhi)
    motor_theta.setRelPosition(stepsTheta)
    while motor_x.isBusy() or motor_y.isBusy() or motor_phi.isBusy() or motor_theta.isBusy():
        sleep(0.2)

def generate_cam1_grid(x_range, y_range, x_steps=8, y_steps=8):
    grid_x = np.linspace(x_range[0], x_range[1], x_steps)
    grid_y = np.linspace(y_range[0], y_range[1], y_steps)
    return list(itertools.product(grid_x, grid_y))

def guided_local_search(cam1_grid, trial_num):
    data_log = []
    for target_x, target_y in cam1_grid:
        print(f"\nTarget cam1 position: ({target_x:.1f}, {target_y:.1f})")

        cam1_pos = cam1.getBeamPosition()
        cam2_pos = cam2.getBeamPosition()

        # Try small random movements to find visibility on both cams
        for trial in range(trial_num):  # max number of attempts
            dx, dy, dphi, dtheta = np.random.uniform(-10, 10, 4)
            move_motors(dx, dy, dphi, dtheta)

            target1 = cam1.getBeamPosition()
            target2 = cam2.getBeamPosition()

            if target1[0] != -1 and target1[1] != -1 and target2[0] != -1 and target2[1] != -1:
                print(f"  Beam visible. cam1: {target1}, cam2: {target2}")
                # TODO: add movement reversal

                # Save the result (you can add target1/2 positions and motor deltas here too)
                data_log.append({
                    "cam1X": cam1_pos[0],
                    "cam1Y": cam1_pos[1],
                    "cam2X": cam2_pos[0],
                    "cam2Y": cam2_pos[1],
                    "t1X": target1[0],
                    "t1Y": target1[1],
                    "t2X": target2[0],
                    "t2Y": target2[1],
                    "stepsX": dx,
                    "stepsY": dy,
                    "stepsPhi": dphi,
                    "stepsTheta": dtheta
                })
                break
        else:
            print("  Failed to find beam for this target.")

    return data_log


def main():
    japc.startSubscriptions()
    sleep(2)

    # Bounding box of cam1 in image coordinates
    cam1_bounds = ((659, 844), (51, 237))
    cam1_grid = generate_cam1_grid(cam1_bounds[0], cam1_bounds[1], x_steps=8, y_steps=8)




if __name__ == "__main__":
    main()