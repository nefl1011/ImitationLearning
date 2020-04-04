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

    if compare_mode == 0:
        file = "expert"
    elif compare_mode == 1:
        file = "ddqn_1mio"
    else:
        file = "ddqn"

    levene = []
    with open("data/conf_dagger/data_%d/levene_results_bool_%s.csv" % (mode, file), "r") as source_file:
        reader = csv.reader(source_file)
        levene = list(reader)

    ks = []
    with open("data/conf_dagger/data_%d/ks_results_bool.csv" % mode, "r") as source_file:
        reader = csv.reader(source_file)
        ks = list(reader)

    for action in range(6):
        print(action)
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

                if ks[agent + 1][action + 1] == "True" and ks[compare_mode + 1][action + 1] == "True":
                    w = "t"
                    l = levene[agent - 2][action + 1] == "True"
                    s1 = stats.ttest_ind(values, compare_values, equal_var=l)[0]
                    s2 = stats.ttest_ind(values, compare_values, equal_var=l)[1]
                elif levene[agent - 2][action + 1] == "True":
                    w = "U"
                    s1 = stats.mannwhitneyu(values, compare_values, alternative='two-sided')[0]
                    s2 = stats.mannwhitneyu(values, compare_values, alternative='two-sided')[1]
                else:
                    w = "chi^2"
                    try:
                        s1 = stats.median_test(values, compare_values)[0]
                        s2 = stats.median_test(values, compare_values)[1]
                    except:
                        s1 = 0
                        s2 = 0
                s = [s1, s2]
                if s[1] < 0.0000000001:
                    sec = "***"
                elif s[1] < 0.0000001:
                    sec = "**"
                elif s[1] < 0.0001:
                    sec = "*"
                else:
                    sec = "%.4f" % np.around(s, decimals=4)[1]
                res = "%s (%.4f %s)" % (sec, np.around(s, decimals=4)[0], w)
                results[agent - 2][action + 1] = res

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

            if ks[agent + 1][7] == "True" and ks[compare_mode + 1][7] == "True":
                w = "t"
                l = levene[agent - 2][7] == "True"
                s1 = stats.ttest_ind(values, compare_values, equal_var=l)[0]
                s2 = stats.ttest_ind(values, compare_values, equal_var=l)[1]
            elif levene[agent - 2][7] == "True":
                w = "U"
                s1 = stats.mannwhitneyu(values, compare_values, alternative='two-sided')[0]
                s2 = stats.mannwhitneyu(values, compare_values, alternative='two-sided')[1]
            else:
                w = "chi^2"
                s1 = stats.median_test(values, compare_values)[0]
                s2 = stats.median_test(values, compare_values)[1]
            s = [s1, s2]
            if s[1] < 0.0000000001:
                sec = "***"
            elif s[1] < 0.0000001:
                sec = "**"
            elif s[1] < 0.0001:
                sec = "*"
            else:
                sec = "%.4f" % np.around(s, decimals=4)[1]
            res = "%s (%.4f %s)" % (sec, np.around(s, decimals=4)[0], w)
            results[agent - 2][7] = res

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
            print(int(float(data[i][compare_mode])))
            compare_values.append(int(float(data[i][compare_mode])))

        for agent in range(3, len(data[0])):
            values = []
            n = 31
            for i in range(1, n):
                values.append(int(float(data[i][agent])))

            if ks[agent + 1][7] == "True" and ks[compare_mode + 1][7] == "True":
                w = "t"
                l = levene[agent - 2][8] == "True"
                s1 = stats.ttest_ind(values, compare_values, equal_var=l)[0]
                s2 = stats.ttest_ind(values, compare_values, equal_var=l)[1]
            elif levene[agent - 2][8] == "True":
                w = "U"
                s1 = stats.mannwhitneyu(values, compare_values, alternative='two-sided')[0]
                s2 = stats.mannwhitneyu(values, compare_values, alternative='two-sided')[1]
            else:
                w = "chi^2"
                s1 = stats.median_test(values, compare_values)[0]
                s2 = stats.median_test(values, compare_values)[1]
            s = [s1, s2]
            if s[1] < 0.0000000001:
                sec = "***"
            elif s[1] < 0.0000001:
                sec = "**"
            elif s[1] < 0.0001:
                sec = "*"
            else:
                sec = "%.4f" % np.around(s, decimals=4)[1]
            res = "%s (%.4f %s)" % (sec, np.around(s, decimals=4)[0], w)
            results[agent - 2][8] = res

    with open("data/conf_dagger/data_%d/test_results_%s.csv" % (mode, file), "x", newline="") as target_file:
        writer = csv.writer(target_file)
        for i in range(len(results)):
            writer.writerow(results[i])

    print(results)

