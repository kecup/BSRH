import pandas as pd
import matplotlib.pyplot as plt

# for com, x and y are swapped!!!

def show_error(df):
    df["dx"] = df["gaussX"] - df["comX"]
    df["dy"] = df["gaussY"] - df["comY"]

    df[["dx", "dy"]].hist(bins=10)
    plt.suptitle("Error Distribution (gauss - com)")
    plt.show()

def show_coord_comparison(df):
    plt.figure(figsize=(6,6))
    plt.scatter(df["gaussX"], df["comX"], alpha=0.6, label="X comparison")
    plt.scatter(df["gaussY"], df["comY"], alpha=0.6, label="Y comparison")
    plt.plot([df[["gaussX","comX","gaussY","comY"]].min().min(),
              df[["gaussX","comX","gaussY","comY"]].max().max()],
             [df[["gaussX","comX","gaussY","comY"]].min().min(),
              df[["gaussX","comX","gaussY","comY"]].max().max()],
             'r--', label="Perfect match")
    plt.xlabel("Gauss Fit")
    plt.ylabel("Center of Mass")
    plt.legend()
    plt.title("Coordinate Comparison (gauss vs com)")
    plt.show()

def main():
    #df = pd.read_csv("Data/test_center_of_mass.csv")3
    df = pd.read_csv("Data/MainCamData.csv")
    print(f"There are {df.shape[0]} rows.")

    # show_error(df)
    show_coord_comparison(df)


if __name__ == "__main__":
    main()