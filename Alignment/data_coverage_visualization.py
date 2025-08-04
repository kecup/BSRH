import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_target_covarage(df, cam_bounds):

    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw rectangle
    width = cam_bounds["x_max"] - cam_bounds["x_min"]
    height = cam_bounds["y_max"] - cam_bounds["y_min"]
    rect_start = (cam_bounds["x_min"], cam_bounds["y_min"])
    rectangle = patches.Rectangle(rect_start, width, height,
                                  linewidth=2, edgecolor='blue', facecolor='none')
    ax.add_patch(rectangle)

    # Scatter points from DataFrame
    ax.scatter(df['cam1X'], df['cam1Y'], color='green', alpha=0.5, label='Cam1')
    ax.scatter(df['cam2X'], df['cam2Y'], color='orange', alpha=0.5, label='Cam2')

    # Formatting
    ax.set_xlabel("cam_x")
    ax.set_ylabel("cam_y")
    ax.legend()
    ax.set_title("Beam Position Coverage")

    plt.grid(True)
    plt.show()


def main():
    cam1_bounds = {"x_min": 659, "x_max": 844, "y_min": 51, "y_max": 237}

    df = pd.read_csv('Data/isr_t3_t4_data_v2.csv')
    df = df[~df[['cam2X', 'cam2Y']].isin([-1]).any(axis=1)]

    visualize_target_covarage(df, cam1_bounds)


if __name__ == "__main__":
    main()