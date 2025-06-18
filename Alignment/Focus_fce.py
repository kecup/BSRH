import pyjapc
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from devices_lib import Camera, Motor, gaussian
from AlignToTarget import Aligner, FeedbackNN, BeamSteerer
import torch
import numpy as np
from scipy.optimize import curve_fit
from time import sleep
import matplotlib.pyplot as plt
import pyrbac


def get_positions(plot_range: tuple[int, int], steps: int) -> list:
    """
    Generates a list of positions in a given range with length steps.

    :param plot_range: Minimum and maximum position of the list.
    :param steps: Number of elements in the list.

    :return: A list of positions.
    """
    positions = np.linspace(plot_range[0], plot_range[1], steps)
    return list(positions)

def wait_for_motor_to_finish(motor: Motor):
    while motor.isBusy():
        sleep(0.2)
        # continue

def focus(motor: Motor, camera: Camera, aligner: Aligner, plot_range: tuple[int, int], steps: int, avg_n_pics: int):
    """
    Focuses a beam.

    :param motor: Motor of the camera to be moved to focus the beam.
    :param camera: Camera to focus the beam on.
    :param aligner: Align beam to the center before focusing.
    :param plot_range: Range of positions to move the motor in.
    :param steps: Number of steps.
    :param avg_n_pics: Number of images to average the gaussian fit over.

    :return: A picture with horizontal and vertical sigma values depending on the motor position.
    """
    # check is beam exists
    amplitude_threshold = 1001 # maybe set this better
    res_x, res_y = camera.getBeamInfo()
    print("init amplitude: " + str(res_x[0]) + " " + str(res_y[0]))
    if res_x[0] < amplitude_threshold or res_y[0] < amplitude_threshold:
        print("There is currently no beam!")
        return [], [], []

    # align to center
    aligner.align(3)
    
    # # get ref (current position of motor)
    # ref = motor.getPosition()
    # # save original configuration
    # original_ref = ref

    # get positions (x-axis values of the graph)
    positions = get_positions(plot_range, steps)

    # move motor to starting point
    start_pos = positions[0] # minimum of given range

    # wait_for_motor_to_finish(motor) # so it doesn't move at the beginning - I don't get it, it stopped working
    motor.setAbsPosition(start_pos)
    wait_for_motor_to_finish(motor)

    # fit gaussian for every position
    sigmas_x = []
    sigmas_y = []
    align_pos = []
    for i in range(len(positions)):
        align_positions = aligner.align(3)
        align_pos.append(align_positions)
        curr_sigmas_x = []
        curr_sigmas_y = []
        for j in range(avg_n_pics):

            image = camera.captureImage()
        
            projection_x = np.sum(image, axis=0)
            g_bounds_x = ([0, 0, 1, 10],[np.inf, len(projection_x), np.inf, np.inf])
            res_x, _ = curve_fit(gaussian, range(len(projection_x)), projection_x, (6e4, np.argmax(projection_x), 10.0, 300.0), bounds = g_bounds_x)
            sigma_x = res_x[2]
        
            # if res_x[0] < amplitude_threshold:
            #     break
            curr_sigmas_x.append(sigma_x)

            projection_y = np.sum(image, axis=1)
            g_bounds_y = ([0, 0, 1, 10],[np.inf, len(projection_y), np.inf, np.inf])
            res_y, _ = curve_fit(gaussian, range(len(projection_y)), projection_y, (6e4, np.argmax(projection_y), 10.0, 300.0), bounds = g_bounds_y)
            sigma_y = res_y[2]
            curr_sigmas_y.append(sigma_y)
            # print(res_x[0], res_y[0])

        sigma_x = np.mean(curr_sigmas_x)
        sigmas_x.append(sigma_x)
        sigma_y = np.mean(curr_sigmas_y)
        sigmas_y.append(sigma_y)

        if i == len(positions) - 1:
            print("position " + str(i+1) + " out of " + str(steps))
            break

        motor.setAbsPosition(positions[i+1])
        wait_for_motor_to_finish(motor)

        sleep(0.2)
        print("position " + str(i+1) + " out of " + str(steps))

    positions = list(positions)
    if len(sigmas_x) == 0:
        print("There is currently no beam!")
        return [], [], []
    if len(positions) != len(sigmas_x): # if there was premature cut-off
        positions = positions[:len(sigmas_x)]

    min_idx_x = sigmas_x.index(min(sigmas_x))
    min_idx_y = sigmas_y.index(min(sigmas_y))
    min_idx = (min_idx_x + min_idx_y) // 2
    # focused_pos = (positions[min_idx_x] + positions[min_idx_y]) // 2
    motor.setAbsPosition(positions[min_idx]) # go to avg min sigma
    aligner.motor_x.setAbsPosition(align_pos[min_idx][0])
    aligner.motor_y.setAbsPosition(align_pos[min_idx][1])
    wait_for_motor_to_finish(motor)

    return positions, sigmas_x, sigmas_y


def graph_focused(positions, sigmas_x, sigmas_y):
    """
    Graph vertical and horizontal sigma values depending on the motor position.
    """
    if positions == [] or sigmas_x == [] or sigmas_y == []:
        print("There is nothing to graph.")
        return
    
    plt.figure(figsize=(8, 5))

    # Plot curves
    plt.plot(positions, sigmas_x, '-o', markersize=3, label='Sigmas X', color='blue', linewidth=2)
    plt.plot(positions, sigmas_y, '-s', markersize=3, label='Sigmas Y', color='red', linewidth=2)

    # Find minima
    min_idx_x = sigmas_x.index(min(sigmas_x))
    min_idx_y = sigmas_y.index(min(sigmas_y))

    # Mark minima points
    plt.plot(positions[min_idx_x], sigmas_x[min_idx_x], 'bo', markersize=5, label='Min X')
    plt.plot(positions[min_idx_y], sigmas_y[min_idx_y], 'rs', markersize=5, label='Min Y')

    # Annotate positions under points
    plt.annotate(f'{positions[min_idx_x]}', 
                xy=(positions[min_idx_x], sigmas_x[min_idx_x]), 
                xytext=(0, -15), textcoords='offset points', 
                ha='center', color='blue')

    plt.annotate(f'{positions[min_idx_y]}', 
                xy=(positions[min_idx_y], sigmas_y[min_idx_y]), 
                xytext=(0, -15), textcoords='offset points', 
                ha='center', color='red')

    plt.xlabel('Positions')
    plt.ylabel('Sigma Values')
    plt.title('Sigma vs Positions of the Motor')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():

    japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
    client = pyrbac.AuthenticationClient()
    token = client.login_kerberos()
    japc.rbacLogin(token=token)
    
    # motor_z = Motor("TST.cfv-375-bsrdev.mot.3-1", japc) # ISR
    # motor_x = Motor('TST.cfv-375-bsrdev.mot.3-4', japc, 168) # ISR
    # motor_y = Motor('TST.cfv-375-bsrdev.mot.3-5', japc, 125) # ISR
    # camera = Camera("TST.BSRH.CAM.CM", japc) # ISR camera
    motor_x = Motor('LHC.BSRH.M3-PHI.B2', japc, 35)  # LHC
    motor_y = Motor('LHC.BSRH.M3-THETA.B2', japc, 30)  # LHC
    motor_z = Motor("LHC.BSRH.M3-Z-TSZ100.B2", japc)  # LHC
    camera = Camera("LHC.BSRH.CAM.CM.B2", japc)  # LHC camera

    target = [484, 304] # Camera center

    model = FeedbackNN(4, 2)
    model.load_state_dict(torch.load('CM_feedbackModel.pth'))
    model.eval()

    steerer = BeamSteerer(model, lr=0.1, iters=30)

    aligner = Aligner(motor_x, motor_y, camera, target, model, steerer)

    japc.startSubscriptions()

    motor_x.setRelPosition(30)
    motor_y.setRelPosition(30)

    positions, sigmas_x, sigmas_y = focus(motor_z, camera, aligner, (40000, 50000), 5, 3)

    japc.stopSubscriptions()
    japc.clearSubscriptions()

    graph_focused(positions, sigmas_x, sigmas_y)

if __name__ == "__main__":
    main()
