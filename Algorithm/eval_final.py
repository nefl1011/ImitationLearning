import csv

if __name__ == '__main__':
    # agent_ids = [78, 82, 88, 89, 198, 297, 298, 311, 312, 317, 318, 319, 329, 355, 404, 416, 434, 435, 447, 448, 485, 486, 487]
    agent_ids = [16, 18, 19, 21, 23, 34, 42, 43]

    iterations = 63
    action = 0
    results = [['0' for _ in range(len(agent_ids) + 3)] for _ in range(iterations)]

    for i in range(len(results[0])):
        val = "Agent_%d" % agent_ids[i - 3]
        if i == 0:
            val = "Expert"
        elif i == 1:
            val = "DDQN_1Mio"
        elif i == 2:
            val = "DDQN"
        results[0][i] = val

    with open("data/conf/log/score.csv", "r") as expert_action_file:
        reader = csv.reader(expert_action_file)
        data = list(reader)
        for i in range(len(data)):
            #if i % 10 == 0:
            #sum = 0
            #for x in range(len(data[i])):
                #sum += int(data[i][x])
            results[i+1][0] = (data[i][action]) # sum

    with open("data/ddqn/log/agent_actions_2/reward_1.csv", "r") as ddqn_action_file:
        reader = csv.reader(ddqn_action_file)
        data = list(reader)
        for i in range(len(data)):
            #sum = 0
            #for x in range(len(data[i])):
                #sum += int(data[i][x])
            results[i+1][1] = (data[i][action]) # sum

    with open("data/ddqn/log/agent_actions_2/reward_3.csv", "r") as ddqn_action_file:
        reader = csv.reader(ddqn_action_file)
        data = list(reader)
        for i in range(len(data)):
            #sum = 0
            #for x in range(len(data[i])):
                #sum += int(data[i][x])
            results[i+1][2] = (data[i][action]) # sum

    for i in range(len(agent_ids)):
        id = agent_ids[i]
        with open("data/conf/log/agent_actions_2/reward_%d.csv" % id, "r") as agent_file:
            reader = csv.reader(agent_file)
            data = list(reader)
            for j in range(len(data)):
                #sum = 0
                #for x in range(len(data[j])):
                    #sum += int(data[j][x])
                results[j + 1][i + 3] = (data[j][action]) # sum

    with open("data/conf_dagger/data_1/reward.csv", "x", newline='') as target_file:
        writer = csv.writer(target_file)
        for i in range(len(results)):
            writer.writerow(results[i])
