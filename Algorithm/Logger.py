import os
import shutil

from Statistic import Statistic

MAX_LOSS = 5
RUN_UPDATE_FREQUENCY = 1
TRAINING_UPDATE_FREQUENCY = 10

class Logger:

    def __init__(self, header, directory_path):
        directory_path = directory_path
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path, ignore_errors=True)
        os.makedirs(directory_path)

        self.score = Statistic("run", "score", RUN_UPDATE_FREQUENCY, directory_path, header)
        self.step = Statistic("run", "step", RUN_UPDATE_FREQUENCY, directory_path, header)
        self.loss = Statistic("update", "loss", RUN_UPDATE_FREQUENCY, directory_path, header)
        self.accuracy = Statistic("update", "accuracy", RUN_UPDATE_FREQUENCY, directory_path, header)
        self.q = Statistic("update", "q", RUN_UPDATE_FREQUENCY, directory_path, header)
        self.t_conf = Statistic("update", "t_conf", RUN_UPDATE_FREQUENCY, directory_path, header)
        self.reward = Statistic("run", "reward", TRAINING_UPDATE_FREQUENCY, directory_path, header)
        self.eval_loss = Statistic("update", "eval_loss", RUN_UPDATE_FREQUENCY, directory_path, header)
        self.eval_acc = Statistic("run", "eval_acc", RUN_UPDATE_FREQUENCY, directory_path, header)

    def add_score(self, score):
        self.score.add_entry(score)

    def add_step(self, step):
        self.step.add_entry(step)

    def add_accuracy(self, accuracy):
        self.accuracy.add_entry(accuracy)

    def add_loss(self, loss):
        loss = min(MAX_LOSS, loss) # for clipping
        self.loss.add_entry(loss)

    def add_q(self, q):
        self.q.add_entry(q)

    def add_t_conf(self, t_conf):
        self.t_conf.add_entry(t_conf)

    def add_reward(self, r):
        self.reward.add_entry(r)

    def add_eval_loss(self, eval_loss):
        self.eval_loss.add_entry(eval_loss)

    def add_eval_acc(self, eval_acc):
        self.eval_acc.add_entry(eval_acc)
