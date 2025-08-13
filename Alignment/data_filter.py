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


def main():
    motor_references = {"x": 40385, "y": 41740, "phi": 139931, "theta": 163051} # isr t3 t4 data collection
    df = pd.read_csv('Data/isr_t3_t4_data_v3_only_phi_theta_10770_samples.csv')

    filtered_df = filter_references(df, motor_references)
    print(f"Original size: {df.shape[0]}, Filtered size: {filtered_df.shape[0]}")

    output_file = "Data/isr_t3_t4_data_v3_only_phi_theta_10770_samples_filtered.csv"

    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered dataset saved to {output_file}")

if __name__ == "__main__":
    main()