import csv
import numpy as np

if __name__ == '__main__':
    # agent_ids = [78, 82, 88, 89, 198, 297, 298, 311, 312, 317, 318, 319, 329, 355, 404, 416, 434, 435, 447, 448, 485, 486, 487]
    agent_ids = [16, 17, 18, 19, 21, 23, 34, 42, 43]

    with open("data/conf_dagger/data_1/eval.csv", "x", newline='') as target_file:
        writer = csv.writer(target_file)
        writer.writerow(
            ['', 'Aktion_0', 'Aktion_1', 'Aktion_2', 'Aktion_3', 'Aktion_4', 'Aktion_5', 'Duration', 'Reward'])

        results = []

        with open("data/conf/log/expert_actions.csv", "r") as expert_file:
            reader = csv.reader(expert_file)
            data = list(reader)
            for i in range(len(data)):
                results = []
                results.append("Expert")
                sum = 0
                for j in range(6):
                    sum += int(data[i][j])
                    results.append(data[i][j])
                results.append(sum)
                with open("data/conf/log/score.csv", "r") as score_file:
                    score_reader = csv.reader(score_file)
                    score_data = list(score_reader)
                    results.append(score_data[i][0])
                writer.writerow(results)

        with open("data/ddqn/log/agent_actions_2/agent_actions1.csv", "r") as ddqn_file:
            reader = csv.reader(ddqn_file)
            data = list(reader)
            for i in range(len(data)):
                results = []
                results.append("DDQN_1Mio")
                sum = 0
                for j in range(6):
                    sum += int(data[i][j])
                    results.append(data[i][j])
                results.append(sum)
                with open("data/ddqn/log/agent_actions_2/reward_1.csv", "r") as score_file:
                    score_reader = csv.reader(score_file)
                    score_data = list(score_reader)
                    results.append(score_data[i][0])
                writer.writerow(results)

        with open("data/ddqn/log/agent_actions_2/agent_actions3.csv", "r") as ddqn_file:
            reader = csv.reader(ddqn_file)
            data = list(reader)
            for i in range(len(data)):
                results = []
                results.append("DDQN")
                sum = 0
                for j in range(6):
                    sum += int(data[i][j])
                    results.append(data[i][j])
                results.append(sum)
                with open("data/ddqn/log/agent_actions_2/reward_3.csv", "r") as score_file:
                    score_reader = csv.reader(score_file)
                    score_data = list(score_reader)
                    results.append(score_data[i][0])
                writer.writerow(results)

        for x in range(len(agent_ids)):
            id = agent_ids[x]
            with open("data/conf/log/agent_actions_2/agent_actions%d.csv" % id, "r") as agent_file:
                reader = csv.reader(agent_file)
                data = list(reader)
                for i in range(len(data)):
                    results = []
                    results.append("Agent_%d" % id)
                    sum = 0
                    for j in range(6):
                        sum += int(data[i][j])
                        results.append(data[i][j])
                    results.append(sum)
                    with open("data/conf/log/agent_actions_2/reward_%d.csv" % id, "r") as score_file:
                        score_reader = csv.reader(score_file)
                        score_data = list(score_reader)
                        results.append(score_data[i][0])
                    writer.writerow(results)
