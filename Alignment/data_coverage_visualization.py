import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_target_coverage_t3_t4(df, cam1_bounds, cam2_bounds):

    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw rectangle for cam1
    width1 = cam1_bounds["x_max"] - cam1_bounds["x_min"]
    height1 = cam1_bounds["y_max"] - cam1_bounds["y_min"]
    rect_start1 = (cam1_bounds["x_min"], cam1_bounds["y_min"])
    rectangle1 = patches.Rectangle(rect_start1, width1, height1,
                                  linewidth=1, edgecolor='blue', facecolor='none')

    # Draw rectangle for cam2
    width2 = cam2_bounds["x_max"] - cam2_bounds["x_min"]
    height2 = cam2_bounds["y_max"] - cam2_bounds["y_min"]
    rect_start2 = (cam2_bounds["x_min"], cam2_bounds["y_min"])
    rectangle2 = patches.Rectangle(rect_start2, width2, height2,
                                  linewidth=1, edgecolor='red', facecolor='none')

    ax.add_patch(rectangle1)
    ax.add_patch(rectangle2)

    # Scatter points from DataFrame
    ax.scatter(df['cam1X'], df['cam1Y'], color='green', alpha=0.5, label='Cam1', s=5)
    ax.scatter(df['cam2X'], df['cam2Y'], color='orange', alpha=0.5, label='Cam2', s=5)

    # ax.scatter(df['t1X'], df['t1Y'], color='green', alpha=0.5, label='Cam1', s=5)
    # ax.scatter(df['t2X'], df['t2Y'], color='orange', alpha=0.5, label='Cam2', s=5)

    # Formatting
    ax.set_xlabel("cam_x")
    ax.set_ylabel("cam_y")
    ax.legend()
    ax.set_title("Beam Position Coverage")

    # plt.grid(True)
    plt.show()

def visualize_target_coverage_t5_t6(df, cam_bounds):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw rectangle for cam
    width = cam_bounds["x_max"] - cam_bounds["x_min"]
    height = cam_bounds["y_max"] - cam_bounds["y_min"]
    rect_start = (cam_bounds["x_min"], cam_bounds["y_min"])
    rectangle = patches.Rectangle(rect_start, width, height,
                                  linewidth=1, edgecolor='blue', facecolor='none')

    ax.add_patch(rectangle)

    # Scatter points from DataFrame
    ax.scatter(df['cam2X'], df['cam2Y'], color='orange', alpha=0.5, label='Cam2', s=5)

    # ax.scatter(df['t2X'], df['t2Y'], color='blue', alpha=0.5, label='Cam2', s=5)

    # Formatting
    ax.set_xlabel("cam_x")
    ax.set_ylabel("cam_y")
    ax.legend()
    ax.set_title("Beam Position Coverage")

    # plt.grid(True)
    plt.show()

def main():
    cam1_bounds = {"x_min": 659, "x_max": 844, "y_min": 51, "y_max": 237}
    cam2_bounds = {"x_min": 0, "x_max": 960, "y_min": 0, "y_max": 640}
    motor_references = {"x": 40385, "y": 41740, "phi": 139931, "theta": 163051}

    df = pd.read_csv('Data/isr_t5_t6_data_v1.csv')

    print(f"There are {df.shape[0]} rows.")


    # visualize_target_coverage_t3_t4(df, cam1_bounds, cam2_bounds)
    visualize_target_coverage_t5_t6(df, cam2_bounds)

    # corr = df[['cam1X', 'cam1Y', 'cam2X', 'cam2Y', 'mX', 'mY', 'mPhi', 'mTheta', 't1X', 't1Y', 't2X', 't2Y',
    #            'stepsX', 'stepsY', 'stepsPhi', 'stepsTheta']].corr()
    # print(corr[['stepsX', 'stepsY', 'stepsPhi', 'stepsTheta']])


if __name__ == "__main__":
    main()