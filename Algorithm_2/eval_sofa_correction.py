import csv

if __name__ == '__main__':
    expert = "../data/Algorithm_2/expert_actions.csv"

    with open(expert, "r") as expert_file:
        expert_reader = csv.reader(expert_file)
        expert_data = list(expert_reader)
        for iter in range(0, 6):
            target = "../data/Algorithm_2/action_%d.csv" % iter
            with open(target, "x", newline='') as target_file:
                writer = csv.writer(target_file)
                length = len(expert_data)
                labels = []
                for i in range(length):
                    if i == 0:
                        labels.append("Id")
                    else:
                        labels.append("Iteration_%d" % i)
                writer.writerow(labels)

                expert_values = []
                expert_values.append(0)
                for i in range(1, length):
                    expert_values.append(expert_data[i][iter])
                writer.writerow(expert_values)

                actions = "data/ddqn/log/action_%d.csv" % iter
                with open(actions, "r") as action_file:
                    action_reader = csv.reader(action_file)
                    action_data = list(action_reader)
                    action_values = []
                    for j in range(len(action_data)):
                        if j == 0:
                            action_values.append(16)
                        elif j == 1:
                            action_values.append(18)
                        elif j == 2:
                            action_values.append(19)
                        elif j == 3:
                            action_values.append(21)
                        elif j == 4:
                            action_values.append(23)
                        elif j == 5:
                            action_values.append(34)
                        elif j == 6:
                            action_values.append(42)
                        else:
                            action_values.append(43)
                        for k in range(length):
                            if len(action_data[j]) > k:
                                action_values.append(action_data[j][k])
                            else:
                                action_values.append(0)
                        writer.writerow(action_values)
                        action_values = []
