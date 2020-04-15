import csv

if __name__ == '__main__':
    for iter in range(0, 6):
        source = "../data/Algorithm_2/action_%d.csv" % iter
        target = "../data/Algorithm_2/action_2_%d.csv" % iter
        with open(source, "r") as source_file:
            reader = csv.reader(source_file)
            with open(target, "x", newline='') as target_file:
                writer = csv.writer(target_file)
                data = list(reader)
                values = []
                values = ["Iteration", "Expert", "Agent_16", "Agent_18", "Agent_19", "Agent_21", "Agent_23", "Agent_34", "Agent_42", "Agent_43"]
                writer.writerow(values)
                values = []
                for i in range(1, len(data[0])):
                    values.append(i)
                    for j in range(1, len(data)):
                        values.append(data[j][i])
                    writer.writerow(values)
                    values = []
