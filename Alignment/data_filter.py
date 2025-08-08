import numpy as np
import pandas as pd


def main():
    df = pd.read_csv('Data/isr_t3_t4_data_only_phi_theta.csv')

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

    print(f"Original size: {len(df)}, Filtered size: {len(filtered_df)}")

    output_file = "Data/isr_t3_t4_data_only_phi_theta_filtered.csv"
    filtered_df = filtered_df.drop(columns=['x_bin', 'y_bin'])

    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered dataset saved to {output_file}")

if __name__ == "__main__":
    main()