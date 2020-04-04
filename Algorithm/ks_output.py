import csv
import math

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

if __name__ == '__main__':
    mode = 1
    if mode == 1:
        agent_ids = [16, 18, 19, 21, 23, 34, 42, 43]
    else:
        agent_ids = [78, 82, 88, 89, 198, 297, 298, 311, 312, 317, 318, 319, 329, 355, 404, 416, 434, 435, 447, 448, 485, 486, 487]

    results = [['0' for _ in range(9)] for _ in range(len(agent_ids) + 4)]

    for i in range(0, len(results) - 1):
        if i == 0:
            val = "Expert (62)"
        elif i == 1:
            val = "DDQN 1Mio (30)"
        elif i == 2:
            val = "DDQN (30)"
        else:
            val = "Agent %d (30)" % agent_ids[i - 3]
        print(val)
        results[i+1][0] = val

    for i in range(1, len(results[0])):
        if i == 0:
            val = ""
        elif i == 1:
            val = "Keine Aktion"
        elif i == 2:
            val = "Schießen"
        elif i == 3:
            val = "Nach Oben"
        elif i == 4:
            val = "Nach Rechts"
        elif i == 5:
            val = "Nach Links"
        elif i == 6:
            val = "Nach Unten"
        elif i == 7:
            val = "Dauer"
        else:
            val = "Belohnung"
        results[0][i] = val

    for action in range(6):
        with open("data/conf_dagger/data_%d/action_%d.csv" % (mode, action), "r") as source_file:
            reader = csv.reader(source_file)
            data = list(reader)

            for agent in range(len(data[0])):
                values = []
                print(data[0][agent])
                if agent == 0:
                    if mode == 1:
                        n = 63
                    else:
                        n = 53
                else:
                    n = 31
                for i in range(1, n):
                    values.append(int(data[i][agent]))

                values_sorted = np.sort(values)

                scaled = (values_sorted - values_sorted.mean())/values_sorted.std()

                normal_numbers = np.random.normal(loc=0, scale=1, size=np.size(scaled))
                normal_numbers = np.sort(normal_numbers)

                ks = stats.kstest(scaled, 'norm')
                if ks[1] < 0.0000000001:
                    sec = "***"
                elif ks[1] < 0.0000001:
                    sec = "**"
                elif ks[1] < 0.0001:
                    sec = "*"
                else:
                    sec = "%.4f" % np.around(ks, decimals=4)[1]
                ks = np.around(ks, decimals=4)
                res = "%s (%.4f)" % (sec, ks[0])
                results[agent + 1][action+1] = res # (np.around(ks, decimals=4)[1] > 0.05 and not(math.isnan(np.around(ks, decimals=4)[1])))

    with open("data/conf_dagger/data_%d/duration.csv" % mode, "r") as source_file:
        reader = csv.reader(source_file)
        data = list(reader)

        for agent in range(len(data[0])):
            values = []

            if agent == 0:
                if mode == 1:
                    n = 63
                else:
                    n = 53
            else:
                n = 31
            for i in range(1, n):
                values.append(int(data[i][agent]))

            values_sorted = np.sort(values)
            scaled = (values_sorted - values_sorted.mean()) / values_sorted.std()

            normal_numbers = np.random.normal(loc=0, scale=1, size=np.size(scaled))
            normal_numbers = np.sort(normal_numbers)

            ks = stats.kstest(scaled, 'norm')
            if ks[1] < 0.0000000001:
                sec = "***"
            elif ks[1] < 0.0000001:
                sec = "**"
            elif ks[1] < 0.0001:
                sec = "*"
            else:
                sec = "%.4f" % np.around(ks, decimals=4)[1]
            ks = np.around(ks, decimals=4)
            res = "%s (%.4f)" % (sec, ks[0])
            results[agent + 1][7] = res # (np.around(ks, decimals=4)[1] > 0.05 and not(math.isnan(np.around(ks, decimals=4)[1])))

    with open("data/conf_dagger/data_%d/reward.csv" % mode, "r") as source_file:
        reader = csv.reader(source_file)
        data = list(reader)

        for agent in range(len(data[0])):
            values = []

            if agent == 0:
                if mode == 1:
                    n = 63
                else:
                    n = 53
            else:
                n = 31
            for i in range(1, n):
                values.append(int(float(data[i][agent])))

            values_sorted = np.sort(values)
            scaled = (values_sorted - values_sorted.mean()) / values_sorted.std()

            normal_numbers = np.random.normal(loc=0, scale=1, size=np.size(scaled))
            normal_numbers = np.sort(normal_numbers)

            ks = stats.kstest(scaled, 'norm')
            if ks[1] < 0.0000000001:
                sec = "***"
            elif ks[1] < 0.0000001:
                sec = "**"
            elif ks[1] < 0.0001:
                sec = "*"
            else:
                sec = "%.4f" % np.around(ks, decimals=4)[1]
            ks = np.around(ks, decimals=4)
            res = "%s (%.4f)" % (sec, ks[0])
            results[agent + 1][8] = res # (np.around(ks, decimals=4)[1] > 0.05 and not(math.isnan(np.around(ks, decimals=4)[1])))

    with open("data/conf_dagger/data_%d/ks_results.csv" % mode, "x", newline="") as target_file:
        writer = csv.writer(target_file)
        for i in range(len(results)):
            writer.writerow(results[i])

    print(results)