import csv

if __name__ == '__main__':
    for iter in range(0, 6):
        source = "../data/Algorithm_1/action_%d.csv" % iter
        target = "../data/Algorithm_1/action_2_%d.csv" % iter
        with open(source, "r") as source_file:
            reader = csv.reader(source_file)
            with open(target, "x", newline='') as target_file:
                writer = csv.writer(target_file)
                data = list(reader)
                values = []
                values = ["Iteration", "Expert", "DDQN_1_Mio", "DDQN", "Agent_78", "Agent_82", "Agent_88", "Agent_89", "Agent_198", "Agent_297", "Agent_298", "Agent_311", "Agent_312", "Agent_317", "Agent_318", "Agent_319", "Agent_329", "Agent_355", "Agent_404", "Agent_416", "Agent_434", "Agent_435", "Agent_447", "Agent_448", "Agent_485", "Agent_486", "Agent_487"]
                writer.writerow(values)
                values = []
                print(len(data))
                print(len(data[0]))
                for i in range(1, len(data[0])):
                    values.append(i)
                    print("i: %d" % i)
                    for j in range(1, len(data)):
                        print("j: %d" % j)
                        values.append(data[j][i])
                    writer.writerow(values)
                    values = []
