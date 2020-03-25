import csv

if __name__ == '__main__':
    path = "data/ddqn/log/agent_actions/"
    mode = "avg"
    for iter in range(0, 6):
        target = "data/ddqn/log/action_%d.csv" % iter
        start = 1
        until = 302

        target_file = open(target, "a", newline='')
        with target_file:

            writer = csv.writer(target_file)
            for i in range(start, until):
                if i == 16 or i == 18 or i == 19 or i == 21 or i == 23 or i == 34 or i == 42 or i == 43:
                    with open(path + "agent_actions%d.csv" % i, "r") as values:
                        reader = csv.reader(values)
                        data = list(reader)
                        results = []
                        for j in range(0, len(data)):
                            results.append(int(data[j][iter]))

                        writer.writerow(results)
