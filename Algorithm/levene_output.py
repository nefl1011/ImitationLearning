import csv
import math

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

if __name__ == '__main__':
    compare_mode = 2
    mode = 2
    if mode == 1:
        agent_ids = [16, 18, 19, 21, 23, 34, 42, 43]
    else:
        agent_ids = [78, 82, 88, 89, 198, 297, 298, 311, 312, 317, 318, 319, 329, 355, 404, 416, 434, 435, 447, 448, 485, 486, 487]

    results = [['0' for _ in range(9)] for _ in range(len(agent_ids) + 1)]

    for i in range(0, len(results) - 1):
        val = "Agent %d (30)" % agent_ids[i]
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

            if compare_mode == 0:
                if mode == 1:
                    c_n = 63
                else:
                    c_n = 53
            else:
                c_n = 31

            compare_values = []

            for i in range(1, c_n):
                compare_values.append(int(data[i][compare_mode]))

            for agent in range(3, len(data[0])):
                values = []
                n = 31
                for i in range(1, n):
                    values.append(int(data[i][agent]))

                l = stats.levene(compare_values, values)
                if l[1] < 0.0000000001:
                    sec = "***"
                elif l[1] < 0.0000001:
                    sec = "**"
                elif l[1] < 0.0001:
                    sec = "*"
                else:
                    sec = "%.4f" % np.around(l, decimals=4)[1]
                l = np.around(l, decimals=4)
                res = "%s (%.4f)" % (sec, l[0])
                results[agent - 2][action + 1] = res # (np.around(l, decimals=4)[1] > 0.05 and not(math.isnan(np.around(l, decimals=4)[1])))

    with open("data/conf_dagger/data_%d/duration.csv" % mode, "r") as source_file:
        reader = csv.reader(source_file)
        data = list(reader)

        if compare_mode == 0:
            if mode == 1:
                c_n = 63
            else:
                c_n = 53
        else:
            c_n = 31

        compare_values = []

        for i in range(1, c_n):
            compare_values.append(int(data[i][compare_mode]))

        for agent in range(3, len(data[0])):
            values = []
            n = 31
            for i in range(1, n):
                values.append(int(data[i][agent]))

            l = stats.levene(compare_values, values)
            if l[1] < 0.0000000001:
                sec = "***"
            elif l[1] < 0.0000001:
                sec = "**"
            elif l[1] < 0.0001:
                sec = "*"
            else:
                sec = "%.4f" % np.around(l, decimals=4)[1]
            l = np.around(l, decimals=4)
            res = "%s (%.4f)" % (sec, l[0])
            results[agent - 2][7] = res # (np.around(l, decimals=4)[1] > 0.05 and not(math.isnan(np.around(l, decimals=4)[1])))

    with open("data/conf_dagger/data_%d/reward.csv" % mode, "r") as source_file:
        reader = csv.reader(source_file)
        data = list(reader)

        if compare_mode == 0:
            if mode == 1:
                c_n = 63
            else:
                c_n = 53
        else:
            c_n = 31

        compare_values = []

        for i in range(1, c_n):
            compare_values.append(int(float(data[i][compare_mode])))

        for agent in range(3, len(data[0])):
            values = []
            n = 31
            for i in range(1, n):
                values.append(int(float(data[i][agent])))

            l = stats.levene(compare_values, values)
            if l[1] < 0.0000000001:
                sec = "***"
            elif l[1] < 0.0000001:
                sec = "**"
            elif l[1] < 0.0001:
                sec = "*"
            else:
                sec = "%.4f" % np.around(l, decimals=4)[1]
            l = np.around(l, decimals=4)
            res = "%s (%.4f)" % (sec, l[0])
            results[agent - 2][8] = res # (np.around(l, decimals=4)[1] > 0.05 and not(math.isnan(np.around(l, decimals=4)[1])))

    if compare_mode == 0:
        file = "expert"
    elif compare_mode == 1:
        file = "ddqn_1mio"
    else:
        file = "ddqn"
    with open("data/conf_dagger/data_%d/levene_results_%s.csv" % (mode, file), "x", newline="") as target_file:
        writer = csv.writer(target_file)
        for i in range(len(results)):
            writer.writerow(results[i])

    print(results)