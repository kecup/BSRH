import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_target_coverage_t1_t2(df, cam1_bounds, cam2_bounds):

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

def visualize_target_coverage_t3_scaled(df, cam_isr_bounds, cam_isr_inner_bounds, cam_lhc_bounds):
    # df =df.head(1)
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw rectangle for isr
    width1 = cam_isr_bounds["x_max"] - cam_isr_bounds["x_min"]
    height1 = cam_isr_bounds["y_max"] - cam_isr_bounds["y_min"]
    rect_start1 = (cam_isr_bounds["x_min"], cam_isr_bounds["y_min"])
    rectangle1 = patches.Rectangle(rect_start1, width1, height1,
                                  linewidth=1, edgecolor='blue', facecolor='none')

    # Draw rectangle for isr inner
    width2 = cam_isr_inner_bounds["x_max"] - cam_isr_inner_bounds["x_min"]
    height2 = cam_isr_inner_bounds["y_max"] - cam_isr_inner_bounds["y_min"]
    rect_start2 = (cam_isr_inner_bounds["x_min"], cam_isr_inner_bounds["y_min"])
    rectangle2 = patches.Rectangle(rect_start2, width2, height2,
                                  linewidth=1, edgecolor='red', facecolor='none')

    # Draw rectangle for lhc
    width3 = cam_lhc_bounds["x_max"] - cam_lhc_bounds["x_min"]
    height3 = cam_lhc_bounds["y_max"] - cam_lhc_bounds["y_min"]
    rect_start3 = (cam_lhc_bounds["x_min"], cam_lhc_bounds["y_min"])
    rectangle3 = patches.Rectangle(rect_start3, width3, height3,
                                  linewidth=1, edgecolor='orange', facecolor='none')

    ax.add_patch(rectangle1)
    ax.add_patch(rectangle2)
    ax.add_patch(rectangle3)

    # Scatter points from DataFrame
    ax.scatter(df['cam1X'], df['cam1Y'], color='pink', alpha=0.5, label='Cam1', s=5)
    #
    # ax.scatter(df['t1X'], df['t1Y'], color='green', alpha=0.5, label='Cam1', s=5)


    # Formatting
    ax.set_xlabel("cam_x")
    ax.set_ylabel("cam_y")
    ax.legend()
    ax.set_title("Beam Position Coverage")

    # plt.grid(True)
    plt.show()

def main():
    cam1_bounds_t1_t2 = {"x_min": 0, "x_max": 0, "y_min": 0, "y_max": 0} # don't know
    cam2_bounds_t1_t2 = {"x_min": 0, "x_max": 0, "y_min": 0, "y_max": 0} # don't know

    cam_isr_bounds_t3 = {"x_min": 659, "x_max": 844, "y_min": 51, "y_max": 237}
    cam_lhc_bounds_t3 = {"x_min": 505, "x_max": 904, "y_min": 40, "y_max": 435}
    cam_isr_inner_bounds = {"x_min": 725, "x_max": 762, "y_min": 112, "y_max": 169}

    cam1_bounds_t3_t4 = {"x_min": 659, "x_max": 844, "y_min": 51, "y_max": 237}
    cam2_bounds_t3_t4 = {"x_min": 0, "x_max": 960, "y_min": 0, "y_max": 640}

    cam1_bounds_t5_t6 = {"x_min": 0, "x_max": 960, "y_min": 0, "y_max": 640} # don't know
    cam2_bounds_t6_t5 = {"x_min": 0, "x_max": 960, "y_min": 0, "y_max": 640} # don't know


    df = pd.read_csv("Data/lhc_t3_t4_data_v4_filtered_scaled2.csv")

    print(f"There are {df.shape[0]} rows.")

    # visualize_target_coverage_t1_t2(df, cam1_bounds_t1_t2, cam2_bounds_t1_t2)
    # visualize_target_coverage_t3_t4(df, cam1_bounds_t3_t4, cam2_bounds_t3_t4)
    # visualize_target_coverage_t5_t6(df, cam2_bounds_t6_t5)

    visualize_target_coverage_t3_scaled(df, cam_isr_bounds_t3, cam_isr_inner_bounds, cam_lhc_bounds_t3)


if __name__ == "__main__":
    main()