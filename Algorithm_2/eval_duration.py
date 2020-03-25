import csv

if __name__ == '__main__':
    path = "data/ddqn/log/agent_actions/"
    mode = "avg"
    target = "data/ddqn/log/duration.csv"
    start = 1
    until = 62

    target_file = open(target, "a", newline='')
    with target_file:

        writer = csv.writer(target_file)
        for i in range(start, until):
            with open(path + "agent_actions%d.csv" % i, "r") as values:
                reader = csv.reader(values)
                data = list(reader)
                results = []
                for k in range(0, len(data)):
                    x = 0
                    for j in range(0, len(data[0])):
                        x += int(data[k][j])
                    results.append(x)
                writer.writerow(results)
