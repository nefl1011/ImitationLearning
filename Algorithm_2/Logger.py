import csv
import os
import shutil

from Statistic import Statistic, Boxplotcurve, DoubleStats, SimpleStats, StackedBarGraph, BarGraph

MAX_LOSS = 5
RUN_UPDATE_FREQUENCY = 1
TRAINING_UPDATE_FREQUENCY = 30

class Logger:

    def __init__(self, header, directory_path):
        self.directory_path = directory_path
        if not os.path.exists(self.directory_path):
            os.makedirs(self.directory_path)

        self.score = SimpleStats("rollouts", "score", RUN_UPDATE_FREQUENCY, directory_path, header)
        self.step = SimpleStats("rollouts", "step", RUN_UPDATE_FREQUENCY, directory_path, header)
        self.loss = DoubleStats("rollouts", "loss", RUN_UPDATE_FREQUENCY, directory_path, header)
        self.accuracy = DoubleStats("rollouts", "accuracy", RUN_UPDATE_FREQUENCY, directory_path, header)
        self.loss_critic = DoubleStats("rollouts", "loss_critic", RUN_UPDATE_FREQUENCY, directory_path, header)
        self.accuracy_critic = DoubleStats("rollouts", "accuracy_critic", RUN_UPDATE_FREQUENCY, directory_path, header)
        self.q = SimpleStats("rollout", "q", RUN_UPDATE_FREQUENCY, directory_path, header)
        self.t_conf = SimpleStats("rollouts", "t_conf_2", RUN_UPDATE_FREQUENCY, directory_path, header)
        self.reward = Boxplotcurve("rollouts", "reward", TRAINING_UPDATE_FREQUENCY, directory_path, header)
        self.expert_action = StackedBarGraph("rollouts", "expert_actions", 18, directory_path, header)
        self.agent_action = BarGraph("rollouts", "agent_actions", 18, directory_path, header)

    def add_score(self, score):
        self.score.add_entry(score)

    def add_step(self, step):
        self.step.add_entry(step)

    def add_accuracy(self, accuracy):
        self.accuracy.add_entry(accuracy)

    def add_loss(self, loss):
        l1 = loss[0]
        l2 = loss[1]  # for clipping
        self.loss.add_entry([l1, l2])

    def add_accuracy_critic(self, accuracy):
        self.accuracy_critic.add_entry(accuracy)

    def add_loss_critic(self, loss):
        l1 = loss[0]
        l2 = loss[1]  # for clipping
        self.loss_critic.add_entry([l1, l2])

    def add_q(self, q):
        self.q.add_entry(q)

    def add_t_conf(self, t_conf):
        self.t_conf.add_entry(t_conf)

    def add_reward(self, r):
        self.reward.add_entry(r)

    def add_expert_action(self, a):
        self.expert_action.add_entry(a)

    def add_agent_action(self, a):
        self.agent_action.add_entry(a)

    def save_expert_action(self):
        self.expert_action.save()

    def save_agent_action(self):
        self.agent_action.save()

    def save_agent_action_avg_std(self):
        self.agent_action.save_avg_std()

    def get_rollouts(self):
        path = self.directory_path + "t_conf.csv"
        rollouts = 0
        if not os.path.exists(path):
            return rollouts
        with open(path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            rollouts = len(data)
        return rollouts
