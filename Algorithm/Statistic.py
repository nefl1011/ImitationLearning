import random
from abc import ABC, abstractmethod
from statistics import mean, stdev, median
import os
import csv
import matplotlib
import numpy as np
from numpy import quantile

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Statistic(ABC):

    def __init__(self, x_label, y_label, update_frequency, directory_path, header):
        self.x_label = x_label
        self.y_label = y_label
        self.update_frequency = update_frequency
        self.directory_path = directory_path
        self.header = header
        self.values = []

    @abstractmethod
    def add_entry(self, value):
        pass

    @abstractmethod
    def _save_png(self, input_path, output_path, small_batch_length, big_batch_length, x_label, y_label):
        pass

    def _save_csv(self, path, score):
        print("Saving " + self.y_label)
        if not os.path.exists(path):
            with open(path, "w"):
                pass
        scores_file = open(path, "a", newline='')
        with scores_file:
            writer = csv.writer(scores_file)
            writer.writerow(score)


class SimpleStats(Statistic):
    def __init__(self, x_label, y_label, update_frequency, directory_path, header):
        super(SimpleStats, self).__init__(
            x_label,
            y_label,
            update_frequency,
            directory_path,
            header)

    def add_entry(self, value):
        self._save_csv(self.directory_path + self.y_label + ".csv", [value])
        self._save_png(input_path=self.directory_path + self.y_label + ".csv",
                       output_path=self.directory_path + self.y_label + ".png",
                       small_batch_length=self.update_frequency,
                       big_batch_length=self.update_frequency * 10,
                       x_label=self.x_label,
                       y_label=self.y_label)

    def _save_png(self, input_path, output_path, small_batch_length, big_batch_length, x_label, y_label):
        x = []
        y = []
        with open(input_path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for i in range(0, len(data)):
                x.append(float(i))
                y.append(float(data[i][0]))

        plt.subplots()
        plt.plot(x, y)

        plt.title(self.header)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # plt.legend(loc="upper left")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()


class DoubleStats(Statistic):
    def __init__(self, x_label, y_label, update_frequency, directory_path, header):
        super(DoubleStats, self).__init__(
            x_label,
            y_label,
            update_frequency,
            directory_path,
            header)

    def add_entry(self, value):
        self._save_csv(self.directory_path + self.y_label + ".csv", value)
        self._save_png(input_path=self.directory_path + self.y_label + ".csv",
                       output_path=self.directory_path + self.y_label + ".png",
                       small_batch_length=self.update_frequency,
                       big_batch_length=self.update_frequency * 10,
                       x_label=self.x_label,
                       y_label=self.y_label)

    def _save_png(self, input_path, output_path, small_batch_length, big_batch_length, x_label, y_label):
        x = []
        y1 = []
        y2 = []
        with open(input_path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for i in range(0, len(data)):
                x.append(float(i))
                y1.append(float(data[i][0]))
                y2.append(float(data[i][1]))

        plt.subplots()
        plt.plot(x, y1, )
        plt.plot(x, y2, color='orange')

        plt.title(self.header)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # plt.legend(loc="upper left")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()


class Boxplotcurve(Statistic):

    def __init__(self, x_label, y_label, update_frequency, directory_path, header):
        super(Boxplotcurve, self).__init__(
            x_label,
            y_label,
            update_frequency,
            directory_path,
            header)

    def add_entry(self, value):
        self.values.append(value)
        if len(self.values) % self.update_frequency == 0:
            mean_value = mean(self.values)
            med = median(self.values)
            std = stdev(self.values)
            minimum = min(self.values)
            quart1 = quantile(self.values, 0.25)
            quart3 = quantile(self.values, 0.75)
            maximum = max(self.values)
            scores = [mean_value, std, minimum, quart1, med, quart3, maximum]
            self._save_csv(self.directory_path + self.y_label + ".csv", scores)
            self._save_png(input_path=self.directory_path + self.y_label + ".csv",
                           output_path=self.directory_path + self.y_label + ".png",
                           small_batch_length=self.update_frequency,
                           big_batch_length=self.update_frequency * 10,
                           x_label=self.x_label,
                           y_label=self.y_label)
            self.values = []

    def _save_png(self, input_path, output_path, small_batch_length, big_batch_length, x_label, y_label):
        x = []
        y = []
        std = []
        minimum = []
        quartil1 = []
        med = []
        quartil3 = []
        maximum = []
        with open(input_path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for i in range(0, len(data)):
                x.append(float(i))
                y.append(float(data[i][0]))
                std.append(float(data[i][1]))
                minimum.append(float(data[i][2]))
                quartil1.append(float(data[i][3]))
                med.append(float(data[i][4]))
                quartil3.append(float(data[i][5]))
                maximum.append(float(data[i][6]))

        plt.subplots()
        plt.plot(x, med, color='blue')
        plt.fill_between(x, med, quartil1, facecolor='cornflowerblue', interpolate=True)
        plt.fill_between(x, med, quartil3, facecolor='cornflowerblue', interpolate=True)
        plt.fill_between(x, quartil1, minimum, facecolor='lightblue', interpolate=True)
        plt.fill_between(x, quartil3, maximum, facecolor='lightblue', interpolate=True)

        plt.title(self.header)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # plt.legend(loc="upper left")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()


class StackedBarGraph(Statistic):

    def __init__(self, x_label, y_label, length, directory_path, header):
        super(StackedBarGraph, self).__init__(
            x_label,
            y_label,
            1,
            directory_path,
            header)
        self.length = length
        self.values = [0] * self.length

    def add_entry(self, value):
        self.values[value] += 1

    def save(self):
        self._save_csv(self.directory_path + self.y_label + ".csv", self.values)
        self._save_png(input_path=self.directory_path + self.y_label + ".csv",
                       output_path=self.directory_path + self.y_label + ".png",
                       small_batch_length=self.update_frequency,
                       big_batch_length=self.update_frequency * 10,
                       x_label=self.x_label,
                       y_label="actions")
        self.values = [0] * self.length

    def _save_png(self, input_path, output_path, small_batch_length, big_batch_length, x_label, y_label):
        actions = []
        with open(input_path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            x = range(1, len(data) + 1)
            for i in range(0, len(data[0])):
                y = []
                for j in range(0, len(data)):
                    y.append(float(data[j][i]))

                actions.append(y)

        totals = []

        def sum_array(array2D, length):
            new_array = []
            for a in range(0, len(array2D[0])):
                c = 0
                for b in range(0, length):
                    c += array2D[b][a]
                new_array.append(c)
            return new_array

        totals = sum_array(actions, len(actions))

        for i in range(0, len(actions)):
            for j in range(0, len(actions[i])):
                actions[i][j] = (actions[i][j] / totals[j]) * 100

        plt.subplots()
        plt.bar(x, actions[0])
        for i in range(1, len(actions)):
            test = sum_array(actions, i)
            plt.bar(x, actions[i], bottom=test)

        plt.title(self.header)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # plt.legend(loc="upper left")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()


class BarGraph(Statistic):

    def __init__(self, x_label, y_label, length, directory_path, header):
        super(BarGraph, self).__init__(
            x_label,
            y_label,
            1,
            directory_path,
            header)
        self.length = length
        self.values = [0] * self.length
        self.values_avg = [0] * self.length
        self.values_std = [0] * self.length
        self.rollouts = 1

    def add_entry(self, value):
        self.values[value] += 1

    def save(self):
        self._save_csv(self.directory_path + self.y_label + "%d.csv" % self.rollouts, self.values)
        self._save_png(input_path=self.directory_path + self.y_label + "%d.csv" % self.rollouts,
                       output_path=self.directory_path + self.y_label + "%d.png" % self.rollouts,
                       small_batch_length=self.update_frequency,
                       big_batch_length=self.update_frequency * 10,
                       x_label=self.x_label,
                       y_label="actions")
        self.values = [0] * self.length

    def save_avg_std(self):
        path = self.directory_path + self.y_label + "%d.csv" % self.rollouts
        actions = []
        with open(path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for i in range(0, len(data)):
                d = []
                for j in range(0, len(data[i])):
                    d.append(float(data[i][j]))
                actions.append(d)

        for i in range(0, len(self.values_avg)):
            vals = []
            for j in range(0, len(actions)):
                vals.append(actions[j][i])
            self.values_avg[i] = np.mean(vals)
            self.values_std[i] = np.std(vals)

        self._save_csv(self.directory_path + self.y_label + "_avg%d.csv" % self.rollouts, self.values_avg)
        self._save_csv(self.directory_path + self.y_label + "_std%d.csv" % self.rollouts, self.values_std)
        self._save_png_avg_std(input_path=self.directory_path + self.y_label + "_avg_std%d.csv" % self.rollouts,
                               output_path=self.directory_path + self.y_label + "_avg_std%d.png" % self.rollouts,
                               small_batch_length=self.update_frequency,
                               big_batch_length=self.update_frequency * 10,
                               x_label=self.x_label,
                               y_label="actions")
        self.values_avg = [0] * self.length
        self.values_std = [0] * self.length
        self.rollouts += 1

    def _save_png(self, input_path, output_path, small_batch_length, big_batch_length, x_label, y_label):
        actions = []
        with open(input_path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            x = range(1, len(data) + 1)
            for i in range(0, len(data[0])):
                y = []
                for j in range(0, len(data)):
                    y.append(float(data[j][i]))

                actions.append(y)

        totals = []

        def sum_array(array2D, length):
            new_array = []
            for a in range(0, len(array2D[0])):
                c = 0
                for b in range(0, length):
                    c += array2D[b][a]
                new_array.append(c)
            return new_array

        totals = sum_array(actions, len(actions))

        for i in range(0, len(actions)):
            for j in range(0, len(actions[i])):
                try:
                    actions[i][j] = (actions[i][j] / totals[j]) * 100
                except ZeroDivisionError:
                    actions[i][j] = 0

        plt.subplots()
        plt.bar(x, actions[0])
        for i in range(1, len(actions)):
            test = sum_array(actions, i)
            plt.bar(x, actions[i], bottom=test)

        plt.title(self.header)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # plt.legend(loc="upper left")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def _save_png_avg_std(self, input_path, output_path, small_batch_length, big_batch_length, x_label, y_label):
        x = range(1, len(self.values_avg) + 1)

        plt.subplots()
        plt.bar(x, self.values_avg, yerr=self.values_std)

        plt.title(self.header)
        plt.xticks(x,
                   ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17'))
        plt.ylabel(y_label)
        # plt.legend(loc="upper left")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
