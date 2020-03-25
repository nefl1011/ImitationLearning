import csv

if __name__ == '__main__':
    path = "data/conf_dagger/log/agent_actions_2/"
    mode = "avg"
    for iter in range(0, 6):
        target = "data/conf_dagger/log/action_%d.csv" % iter
        start = 1
        until = 513

        target_file = open(target, "a", newline='')
        with target_file:

            writer = csv.writer(target_file)
            for i in range(start, until):
                if i == 74 or i == 78 or i == 82 or i == 88 or i == 89 or i == 177 or i == 194 or i == 198 or i == 201 or i == 248 or i == 249 or i == 257 or i == 258 or i == 261 or i == 296 or i == 298 or i == 300:
                    with open(path + "agent_actions_%d.csv" % i, "r") as values:
                        reader = csv.reader(values)
                        data = list(reader)
                        results = []
                        for j in range(0, len(data)):
                            results.append(int(data[j][iter]))

                        writer.writerow(results)
