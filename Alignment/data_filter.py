import numpy as np
import pandas as pd

def filter_bins(df):
    x_bins_num = 20
    y_bins_num = 20
    # Define grid for Cam2
    x_bins = np.linspace(0, 968, x_bins_num)
    y_bins = np.linspace(0, 608, y_bins_num)

    # Assign each point to a bin
    df['x_bin'] = np.digitize(df['cam2X'], x_bins)
    df['y_bin'] = np.digitize(df['cam2Y'], y_bins)

    # Group and sample max N per bin
    max_samples_per_bin = 35  # Adjust based on desired dataset size
    filtered_df = (
        df.groupby(['x_bin', 'y_bin'], group_keys=False)
        .apply(lambda g: g.sample(min(len(g), max_samples_per_bin), random_state=42))
    )
    filtered_df = filtered_df.drop(columns=['x_bin', 'y_bin'])

    return filtered_df


def filter_references(df, motor_references):
    """
    Filter out all data from references as backlash creates 4 individual clusters.
    """
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df[~(df == -1.0).any(axis=1)] # for the accidental -1 values
    df = df[~(df == df.columns).all(axis=1)] # for the accidental header repeat
    print(f"There are {df.shape[0]} rows.")

    # df_references = df[(df['mX'] != motor_references['x']) & (df['mY'] != motor_references['y']) &
    #                    (df['mPhi'] != motor_references['phi']) & (df['mTheta'] != motor_references['theta'])]

    df_references = df[(df['mPhi'] != motor_references['phi']) & (df['mTheta'] != motor_references['theta'])]

    # df_references = df[(df['mX'] != motor_references['x']) & (df['mY'] != motor_references['y'])]

    return df_references

def filter_outliers(df, cam2_bounds, cam1_bounds=None):

    df_filtered2 = df[
        (df['cam2X'] >= cam2_bounds['x_min']) & (df['cam2X'] <= cam2_bounds['x_max']) &
        (df['cam2Y'] >= cam2_bounds['y_min']) & (df['cam2Y'] <= cam2_bounds['y_max'])]

    if cam1_bounds != None:
        df_filtered1 = df[
            (df_filtered2['cam1X'] >= cam1_bounds['x_min']) & (df_filtered2['cam1X'] <= cam1_bounds['x_max']) &
            (df_filtered2['cam1Y'] >= cam1_bounds['y_min']) & (df_filtered2['cam1Y'] <= cam1_bounds['y_max'])]

        return df_filtered1

    return df_filtered2

def filter_center_targets(df, bounds):
    filtered_df = df[
        (df["t2X"].between(bounds["x_min"], bounds["x_max"])) &
        (df["t2Y"].between(bounds["y_min"], bounds["y_max"]))
        ]

    return filtered_df

def main():
    cam1_bounds = {"x_min": 659, "x_max": 844, "y_min": 51, "y_max": 237}
    cam2_bounds = {"x_min": 0, "x_max": 960, "y_min": 0, "y_max": 640}
    cam2_central = {"x_min": 400, "x_max": 560, "y_min": 270, "y_max": 370}
    motor_references_t3_t4_v3 = {"x": 40385, "y": 41740, "phi": 139931, "theta": 163051} # isr t3 t4 data collection
    motor_references_t3_t4_v4 = {"x": 40385, "y": 41740, "phi": 140381, "theta": 163301}
    motor_references_t5_t6 = {"phi": 13444, "theta": 13360} # isr t5 t6 data collection v1

    df = pd.read_csv('Data/isr_t3_t4_data_v4.csv')

    # filtered_df = filter_outliers(df, cam2_bounds)
    filtered_df = filter_references(df, motor_references_t3_t4_v4)
    # filtered_df = filter_center_targets(filtered_df, cam2_central)

    print(f"Original size: {df.shape[0]}, Filtered size: {filtered_df.shape[0]}")

    output_file = "Data/isr_t3_t4_data_v4_filtered.csv"

    # filtered_cropped_df = filtered_df.head(7500)
    # print(f"Original size: {df.shape[0]}, Filtered size: {filtered_cropped_df.shape[0]}")

    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered dataset saved to {output_file}")

if __name__ == "__main__":
    main()