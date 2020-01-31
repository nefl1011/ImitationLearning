import os
import shutil

from Statistic import Statistic, Boxplotcurve, DoubleStats, SimpleStats

MAX_LOSS = 5
RUN_UPDATE_FREQUENCY = 1
TRAINING_UPDATE_FREQUENCY = 10

class Logger:

    def __init__(self, header, directory_path):
        directory_path = directory_path
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path, ignore_errors=True)
        os.makedirs(directory_path)

        self.score = SimpleStats("run", "score", RUN_UPDATE_FREQUENCY, directory_path, header)
        self.step = SimpleStats("run", "step", RUN_UPDATE_FREQUENCY, directory_path, header)
        self.loss = DoubleStats("update", "loss", RUN_UPDATE_FREQUENCY, directory_path, header)
        self.accuracy = DoubleStats("update", "accuracy", RUN_UPDATE_FREQUENCY, directory_path, header)
        self.q = SimpleStats("update", "q", RUN_UPDATE_FREQUENCY, directory_path, header)
        self.t_conf = SimpleStats("update", "t_conf", RUN_UPDATE_FREQUENCY, directory_path, header)
        self.reward = Boxplotcurve("run", "reward", TRAINING_UPDATE_FREQUENCY, directory_path, header)

    def add_score(self, score):
        self.score.add_entry(score)

    def add_step(self, step):
        self.step.add_entry(step)

    def add_accuracy(self, accuracy):
        self.accuracy.add_entry(accuracy)

    def add_loss(self, loss):
        l1 = min(MAX_LOSS, loss[0])
        l2 = min(MAX_LOSS, loss[1])  # for clipping
        self.loss.add_entry([l1, l2])

    def add_q(self, q):
        self.q.add_entry(q)

    def add_t_conf(self, t_conf):
        self.t_conf.add_entry(t_conf)

    def add_reward(self, r):
        self.reward.add_entry(r)
