import dill as pickle
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
     # so first_images contains the last images without occulter, so we can then calculate the BG by performing a 2D deconvolution...
    #graph_cpc(results)

    f = open("BG_results_v2.pkl", 'rb')
    results = pickle.load(f)
    f.close()

    bg = {}

    key = list(results.keys())
    for i in range(len(results)):
        bunchNumbers = list(results[key[i]].keys())
        bunchNumbers = sorted(bunchNumbers)
        if i == 0:
            for j in range(len(bunchNumbers)):
                bg.setdefault(bunchNumbers[j], [[], [], [], [], []])
        for bN in bunchNumbers:
            time_Avg = results[key[i]][bN]['time'][0]

            bg_x_Avg = np.mean(results[key[i]][bN]['bg_x'])
            bg_y_Avg = np.mean(results[key[i]][bN]['bg_y'])

            bg_x_Std = np.std(results[key[i]][bN]['bg_x'])
            bg_y_Std = np.std(results[key[i]][bN]['bg_y'])

            bg[bN][0].append(time_Avg)
            bg[bN][1].append(bg_x_Avg)
            bg[bN][2].append(bg_x_Std)
            bg[bN][3].append(bg_y_Avg)
            bg[bN][4].append(bg_y_Std)
    g = open("AvgBG.pkl", 'wb')
    pickle.dump(bg, g)
    g.close()

    key = list(bg.keys())
    for i in key:
        plt.figure(1)
        plt.errorbar(bg[i][0], bg[i][1], bg[i][2], label = str(i))

        plt.figure(2)
        plt.errorbar(bg[i][0], bg[i][3], bg[i][4], label=str(i))
    plt.figure(1)
    plt.legend()

    plt.figure(2)
    plt.legend()

    plt.show()
