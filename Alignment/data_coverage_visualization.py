import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_target_covarage(df, cam1_bounds, cam2_bounds):

    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw rectangle for cam1
    width = cam1_bounds["x_max"] - cam1_bounds["x_min"]
    height = cam1_bounds["y_max"] - cam1_bounds["y_min"]
    rect_start = (cam1_bounds["x_min"], cam1_bounds["y_min"])
    rectangle1 = patches.Rectangle(rect_start, width, height,
                                  linewidth=1, edgecolor='blue', facecolor='none')

    # Draw rectangle for cam2
    width = cam2_bounds["x_max"] - cam2_bounds["x_min"]
    height = cam2_bounds["y_max"] - cam2_bounds["y_min"]
    rect_start = (cam2_bounds["x_min"], cam2_bounds["y_min"])
    rectangle2 = patches.Rectangle(rect_start, width, height,
                                  linewidth=1, edgecolor='red', facecolor='none')

    ax.add_patch(rectangle1)
    ax.add_patch(rectangle2)

    # Scatter points from DataFrame
    ax.scatter(df['cam1X'], df['cam1Y'], color='green', alpha=0.5, label='Cam1', s=5)
    ax.scatter(df['cam2X'], df['cam2Y'], color='orange', alpha=0.5, label='Cam2', s=5)

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

    df = pd.read_csv('Data/isr_t3_t4_data_only_xy.csv')
    df = df[~df[['cam2X', 'cam2Y']].isin([-1]).any(axis=1)]

    print(f"There are {df.shape[0]} samples")

    visualize_target_covarage(df, cam1_bounds, cam2_bounds)
    # corr = df[['cam1X', 'cam1Y', 'cam2X', 'cam2Y', 'mX', 'mY', 'mPhi', 'mTheta', 't1X', 't1Y', 't2X', 't2Y',
    #            'stepsX', 'stepsY', 'stepsPhi', 'stepsTheta']].corr()
    # print(corr[['stepsX', 'stepsY', 'stepsPhi', 'stepsTheta']])


if __name__ == "__main__":
    main()