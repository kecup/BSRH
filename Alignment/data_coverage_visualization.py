import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Data/isr_t3_t4_data.csv')
df = df[~df[['cam2X', 'cam2Y']].isin([-1]).any(axis=1)]

plt.scatter(df['cam1X'], df['cam1Y'], label="Cam1", alpha=0.5)
plt.scatter(df['cam2X'], df['cam2Y'], label="Cam2", alpha=0.5)
plt.title("Beam Position Coverage")
plt.legend()
plt.show()