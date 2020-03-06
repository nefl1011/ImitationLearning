import csv

if __name__ == '__main__':
    path = "data/conf_dagger/log/agent_actions_2/"
    mode = "avg"
    target = "data/conf_dagger/log/agent_%s_all.csv" % mode
    start = 1
    until = 302

    target_file = open(target, "a", newline='')
    with target_file:
        for i in range(start, until):
            writer = csv.writer(target_file)
            with open(path + "agent_actions_%s%d.csv" % (mode, i), "r") as values:
                reader = csv.reader(values)
                data = list(reader)
                x = []
                for j in range(0, len(data[0])):
                    x.append(float(data[0][j]))

            writer.writerow(x)
