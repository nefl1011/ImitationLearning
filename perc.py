import csv

if __name__ == '__main__':
	agent_max = 513
	results = [[0 for _ in range(agent_max)] for _ in range(18)]
	
	for i in range(1, agent_max):
		with open("Algorithm/data/conf_dagger/log/agent_actions_2/agent_actions%d.csv" % i, "r") as action_file:
			reader = csv.reader(action_file)
			data = list(reader)
			
			for j in range(18):
				sum = 0
				for k in range(len(data)):
					print()
					sum += int(data[k][j])
				results[j][i-1] = sum

	# print(results)
	
	with open("perc2.csv", "x", newline='') as target_file:
		writer = csv.writer(target_file)
		for i in range(len(results)):
			writer.writerow(results[i])