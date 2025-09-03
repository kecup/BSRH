import pandas as pd
import numpy as np


def synthesize_outside_data(df, isr, k_list=[1.25, 2.0, 3.0, 3.5, 4.0, 5.0, 6.0, 6.5, 7.0]):
    """
    Expand dataset by generating synthetic cam positions outside the bounding box.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ["cam1X","cam1Y","t1X","t1Y","stepsPhi","stepsTheta"]
    canvas_x : (min, max)
        Allowed X range for camera
    canvas_y : (min, max)
        Allowed Y range for camera
    k_list : list of float
        Scaling factors > 1 to push camera further away

    Returns
    -------
    pd.DataFrame
        Augmented dataset with synthetic rows
    """
    if isr:
        canvas_x=(659, 844)
        canvas_y=(51, 237)
    else:
        canvas_x = (505, 904)
        canvas_y = (40, 435)

    rows = []
    for idx, row in df.iterrows():
        dx = row["cam1X"] - row["t1X"]
        dy = row["cam1Y"] - row["t1Y"]

        for k in k_list:
            # push camera further away from target
            new_camX = row["t1X"] + k * dx
            new_camY = row["t1Y"] + k * dy

            # clip to stay within canvas
            new_camX = np.clip(new_camX, *canvas_x)
            new_camY = np.clip(new_camY, *canvas_y)

            if (new_camX in canvas_x) or (new_camY in canvas_y):
                continue

            new_row = {
                "cam1X": new_camX,
                "cam1Y": new_camY,
                "t1X": row["t1X"],
                "t1Y": row["t1Y"],
                "stepsPhi": k * row["stepsPhi"],
                "stepsTheta": k * row["stepsTheta"],
                "orig_index": idx,
                "scale_factor": k
            }
            rows.append(new_row)

    return pd.DataFrame(rows)

def main():
    # for isr: canvas_x=(659, 844), canvas_y=(51, 237)
    # for lhc: canvas_x=(505, 904), canvas_y=(40, 435)
    df = pd.read_csv("Data/isr_t3_t4_data_v4_filtered.csv")
    df_aug = synthesize_outside_data(df, isr=False)

    output_file = "Data/lhc_t3_t4_data_v4_filtered_scaled2.csv"

    df_aug.to_csv(output_file, index=False)
    print(f"Filtered dataset saved to {output_file}")


if __name__ == "__main__":
    main()