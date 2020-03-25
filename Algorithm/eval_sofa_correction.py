import csv

if __name__ == '__main__':
    expert = "../data/Algorithm_1/expert_actions.csv"

    with open(expert, "r") as expert_file:
        expert_reader = csv.reader(expert_file)
        expert_data = list(expert_reader)
        for iter in range(0, 6):
            target = "../data/Algorithm_1/action_%d.csv" % iter
            with open(target, "w", newline='') as target_file:
                writer = csv.writer(target_file)
                length = len(expert_data)
                labels = []
                for i in range(length + 2):
                    if i == 0:
                        labels.append("Id")
                    else:
                        labels.append("Iteration_%d" % i)
                writer.writerow(labels)

                expert_values = []
                expert_values.append(0)
                for i in range(1, length + 2):
                    if i >= length:
                        expert_values.append(0)
                    else:
                        expert_values.append(expert_data[i][iter + 1])
                writer.writerow(expert_values)

                actions = "data/conf_dagger/log/action_%d.csv" % iter
                with open(actions, "r") as action_file:
                    action_reader = csv.reader(action_file)
                    action_data = list(action_reader)
                    action_values = []
                    for j in range(len(action_data)):
                        if j == 0:
                            action_values.append(74)
                        elif j == 1:
                            action_values.append(78)
                        elif j == 2:
                            action_values.append(82)
                        elif j == 3:
                            action_values.append(88)
                        elif j == 4:
                            action_values.append(89)
                        elif j == 5:
                            action_values.append(177)
                        elif j == 6:
                            action_values.append(194)
                        elif j == 7:
                            action_values.append(198)
                        elif j == 8:
                            action_values.append(201)
                        elif j == 9:
                            action_values.append(248)
                        elif j == 10:
                            action_values.append(249)
                        elif j == 11:
                            action_values.append(257)
                        elif j == 12:
                            action_values.append(258)
                        elif j == 13:
                            action_values.append(261)
                        elif j == 14:
                            action_values.append(296)
                        elif j == 15:
                            action_values.append(298)
                        else:
                            action_values.append(300)
                        for k in range(len(action_data[0])):
                            action_values.append(action_data[j][k])
                        writer.writerow(action_values)
                        action_values = []
