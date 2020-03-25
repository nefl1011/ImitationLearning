import csv
import numpy as np

if __name__ == '__main__':
    actions_path = "../data/Algorithm_1/agent_actions_2/"
    reward_path = "../data/Algorithm_1/reward/"
    expert = "../data/Algorithm_1/expert_actions.csv"
    expert_scores = "../data/Algorithm_1/score.csv"

    ddqn_all_actions = "../data/Algorithm_1/ddqn_all_actions.csv"
    ddqn_1_actions = "../data/Algorithm_1/ddqn_1_actions.csv"
    ddqn_all_rewards = "../data/Algorithm_1/ddqn_all_rewards.csv"
    ddqn_1_rewards = "../data/Algorithm_1/ddqn_1_rewards.csv"

    target = "data/conf_dagger/log/all.csv"
    start = 1
    # until 62
    until = 302

    target_file = open(target, "x", newline='')
    with target_file:

        writer = csv.writer(target_file)

        # expert first
        with open(expert, "r") as expert_values:
            expert_reader = csv.reader(expert_values)
            expert_data = list(expert_reader)
            for i in range(len(expert_data)):
                if i % 10 == 0 and (i != 120 and i != 130 and i != 220):
                    expert_results = []
                    expert_results.append("Expert")
                    for j in range(len(expert_data[i])):
                        expert_results.append(expert_data[i][j])
                    expert_results.append(np.sum(expert_data[i]))
                    with open(expert_scores) as score_values:
                        score_reader = csv.reader(score_values)
                        score_data = list(score_reader)
                        expert_results.append(int(score_data[i]))
                    writer.writerow(expert_results)

        with open(ddqn_all_actions, "r") as ddqn_all_values:
            ddqn_all_reader = csv.reader(ddqn_all_values)
            ddqn_all_data = list(ddqn_all_reader)
            for i in range(len(ddqn_all_data)):
                ddqn_all_results = []
                ddqn_all_results.append("DDQN_1Mio")
                for j in range(len(ddqn_all_data[i])):
                    expert_results.append(ddqn_all_data[i][j])
                ddqn_all_results.append(np.sum(ddqn_all_data[i]))
                with open(ddqn_all_rewards) as ddqn_all_score_values:
                    ddqn_all_score_reader = csv.reader(ddqn_all_score_values)
                    ddqn_all_score_data = list(ddqn_all_score_reader)
                    ddqn_all_results.append(int(ddqn_all_score_data[i]))
                writer.writerow(ddqn_all_results)

        with open(ddqn_1_actions, "r") as ddqn_1_values:
            ddqn_1_reader = csv.reader(ddqn_1_values)
            ddqn_1_data = list(ddqn_1_reader)
            for i in range(len(ddqn_1_data)):
                ddqn_1_results = []
                ddqn_1_results.append("DDQN_1")
                for j in range(len(ddqn_1_data[i])):
                    expert_results.append(ddqn_1_data[i][j])
                ddqn_1_results.append(np.sum(ddqn_1_data[i]))
                with open(ddqn_1_rewards) as ddqn_1_score_values:
                    ddqn_1_score_reader = csv.reader(ddqn_1_score_values)
                    ddqn_1_score_data = list(ddqn_1_score_reader)
                    ddqn_1_results.append(int(ddqn_1_score_data[i]))
                writer.writerow(ddqn_1_results)

        for i in range(start, until):
            if i == 74 or i == 78 or i == 82 or i == 88 or i == 89 or i == 177 or i == 194 or i == 198 or i == 201 or i == 248 or i == 249 or i == 257 or i == 258 or i == 261 or i == 296 or i == 298 or i == 300:
            # if i == 16 or i == 18 or i == 21 or i == 23 or i == 34 or i == 42 or i == 43:
                with open(actions_path + "agent_actions_%d.csv" % i, "r") as values:
                    reader = csv.reader(values)
                    data = list(reader)
                    for j in range(0, len(data)):
                        results = []
                        results.append("Agent_%d" % i)
                        for k in range(len(data[j])):
                            results.append(data[j][k])
                        results.append(np.sum(data[j]))
                        with open(reward_path) as reward_values:
                            reward_reader = csv.reader(reward_values)
                            reward_data = list(reward_reader)
                            expert_results.append(int(reward_data[j]))
                        writer.writerow(results)
