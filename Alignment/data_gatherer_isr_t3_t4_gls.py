# gather data on isr - mirror2 (x, y, phi, theta) + target3 (see-through) + target4
# use guided local search to cover the region of cam1
import pandas as pd
import pyjapc
import csv
import random
from time import sleep
import sys
import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

# Configuration
CAM1_BOUNDS = {"x_min": 659, "x_max": 844, "y_min": 51, "y_max": 237}
GRID_STEPS_X = 8
GRID_STEPS_Y = 8
MAX_TRIALS = 200
STEP_SIZE = 50
MOTOR_REFERENCES = {"x": 40085, "y": 41740, "phi": 141900, "theta": 161000} # to be measured, they keep changing with time
# POSITION_TOLERANCE = 10  # pixels
CSV_FILENAME = "isr_t3_t4_gls_data.csv"

def move_to_references():
    """
    Moves motors to pre-measured references.
    """
    motor_x.setAbsPosition(MOTOR_REFERENCES["x"])
    motor_y.setAbsPosition(MOTOR_REFERENCES["y"])
    motor_phi.setAbsPosition(MOTOR_REFERENCES["phi"])
    motor_theta.setAbsPosition(MOTOR_REFERENCES["theta"])
    while motor_x.isBusy() or motor_y.isBusy() or motor_phi.isBusy() or motor_theta.isBusy():
        sleep(0.2)

def move_motors(stepsX: int, stepsY: int, stepsPhi: int, stepsTheta: int):
    """
    Moves motors by given deltas.
    """
    motor_x.setRelPosition(stepsX)
    motor_y.setRelPosition(stepsY)
    motor_phi.setRelPosition(stepsPhi)
    motor_theta.setRelPosition(stepsTheta)
    while motor_x.isBusy() or motor_y.isBusy() or motor_phi.isBusy() or motor_theta.isBusy():
        sleep(0.2)

def reverse_motors(stepsX: int, stepsY: int, stepsPhi: int, stepsTheta: int):
    """
    Reverses motor movements by given deltas.
    """
    motor_x.setRelPosition(-2*stepsX) # 2x because reversal often fails for some reason
    motor_y.setRelPosition(-2*stepsY)
    motor_phi.setRelPosition(-2*stepsPhi)
    motor_theta.setRelPosition(-2*stepsTheta)
    while motor_x.isBusy() or motor_y.isBusy() or motor_phi.isBusy() or motor_theta.isBusy():
        sleep(0.2)

def generate_cam1_grid():
    """
    Generates a grid covering target 1 area of camera 1. (t3)
    """
    x_vals = np.linspace(CAM1_BOUNDS["x_min"], CAM1_BOUNDS["x_max"], GRID_STEPS_X)
    y_vals = np.linspace(CAM1_BOUNDS["y_min"], CAM1_BOUNDS["y_max"], GRID_STEPS_Y)

    grid = []
    for x, y in itertools.product(x_vals, y_vals):
        if (x == CAM1_BOUNDS["x_min"] or x == CAM1_BOUNDS["x_max"] or
                y == CAM1_BOUNDS["y_min"] or y == CAM1_BOUNDS["y_max"]):
            continue  # Skip border points
        grid.append((x, y))

    return grid


def cam1_distance(pos, target):
    """
    Calculates distance of current position to target position.
    """
    return np.linalg.norm(np.array(pos) - np.array(target))


def align_to_target(target_pos):
    """
    Randomly generates motor movements and tries to get closer to target on camera 1 grid.
    """

    cam1_pos = cam1.getBeamPosition()
    cam2_pos = cam2.getBeamPosition()

    if cam1_pos[0] == -1 or cam1_pos[1] == -1 or cam2_pos[0] == -1 or cam2_pos[1] == -1: # init check
        move_to_references()
        cam1_pos = cam1.getBeamPosition()
        cam2_pos = cam2.getBeamPosition()
        if cam1_pos[0] == -1 or cam1_pos[1] == -1 or cam2_pos[0] == -1 or cam2_pos[1] == -1:
            print("Cannot find beam at references.")
            return []
    
    motors = [motor_x, motor_y, motor_phi, motor_theta]
    prev_dist = cam1_distance(cam1_pos, target_pos)
    print(f"Current distance is {prev_dist:.2f}")
    results = []
 
    for _ in range(MAX_TRIALS):
        print("iteration " + str(_+1))
        print(f"Current distance is {prev_dist:.2f}")
        deltas = {k: 0.0 for k in motors}
        chosen = random.sample(list(motors), random.randint(1, 2))
        for k in chosen:
            deltas[k] = int(np.random.uniform(-STEP_SIZE, STEP_SIZE))

        cam1_pos = cam1.getBeamPosition()
        cam2_pos = cam2.getBeamPosition()

        if cam1_pos[0] == -1 or cam1_pos[1] == -1 or cam2_pos[0] == -1 or cam2_pos[1] == -1: # safety check if beam lost
            print("Reversal did not work, sending to references.")
            move_to_references()
            cam1_pos = cam1.getBeamPosition()
            prev_dist = cam1_distance(cam1_pos, target_pos)
            continue

        move_motors(deltas[motor_x], deltas[motor_y], deltas[motor_phi], deltas[motor_theta])
        cam1_new = cam1.getBeamPosition()
        cam2_new = cam2.getBeamPosition()

        if cam1_new[0] == -1 or cam1_new[1] == -1 or cam2_new[0] == -1 or cam2_new[1] == -1:
            print("reversing motors")
            reverse_motors(deltas[motor_x], deltas[motor_y], deltas[motor_phi], deltas[motor_theta])
            continue

        new_dist = cam1_distance(cam1_new, target_pos)

        # want to log all feasible data
        data = [cam1_pos[0], cam1_pos[1], cam2_pos[0], cam2_pos[1], cam1_new[0], cam1_new[1], cam2_new[0], cam2_new[1],
                deltas[motor_x], deltas[motor_y], deltas[motor_phi], deltas[motor_theta]]
        results.append(data)
        
        if new_dist < prev_dist:
            # Good move: keep it
            prev_dist = new_dist
            print(f"Improved distance to {new_dist:.2f}")
        else:
            reverse_motors(deltas[motor_x], deltas[motor_y], deltas[motor_phi], deltas[motor_theta])

    return results

def visualize_target_covarage(target, data):
    df = pd.DataFrame(data, columns=["cam1X", "cam1Y", "cam2X", "cam2Y", "t1X", "t1Y", "t2X", "t2Y", "stepsX",
              "stepsY", "stepsPhi", "stepsTheta"])

    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw rectangle
    width = CAM1_BOUNDS["x_max"] - CAM1_BOUNDS["x_min"]
    height = CAM1_BOUNDS["y_max"] - CAM1_BOUNDS["y_min"]
    rect_start = (CAM1_BOUNDS["x_min"], CAM1_BOUNDS["y_min"])
    rectangle = patches.Rectangle(rect_start, width, height,
                                  linewidth=2, edgecolor='blue', facecolor='none')
    ax.add_patch(rectangle)

    # Plot target point
    ax.scatter(*target, color='red', s=100, marker='x', label='Target')

    # Scatter points from DataFrame
    ax.scatter(df['cam1X'], df['cam1Y'], color='green', alpha=0.7, label='Camera Points')

    # Formatting
    ax.set_xlim(600, 900)
    ax.set_ylim(0, 300)
    ax.set_xlabel("cam1x")
    ax.set_ylabel("cam1y")
    ax.legend()
    ax.set_title("Rectangle, Target Point, and Camera Data")

    plt.grid(True)
    plt.show()


def main():
    japc.startSubscriptions()
    sleep(2)

    # Bounding box of cam1 in image coordinates
    cam1_grid = generate_cam1_grid()

    move_to_references() # move motors to visible references

    header = ["cam1X", "cam1Y", "cam2X", "cam2Y", "mTheta", "t1X", "t1Y", "t2X", "t2Y", "stepsX",
              "stepsY", "stepsPhi", "stepsTheta"]
    file_exists = os.path.isfile(CSV_FILENAME)

    with open(CSV_FILENAME, 'a') as file:
        file_writer = csv.writer(file)

        if not file_exists:
            file_writer.writerow(header)

        for tx, ty in cam1_grid:
            print(f"\nAligning to target: ({tx:.1f}, {ty:.1f})")
            data = align_to_target((tx, ty))
            visualize_target_covarage((tx, ty), data)
            file_writer.writerows(data)
            print(f"Number of data for current target is {len(data)}")
            

if __name__ == "__main__":
    main()