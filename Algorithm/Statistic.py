from statistics import mean, stdev, median
import os
import csv
import matplotlib
from numpy import quantile

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Statistic:

    def __init__(self, x_label, y_label, update_frequency, directory_path, header):
        self.x_label = x_label
        self.y_label = y_label
        self.update_frequency = update_frequency
        self.directory_path = directory_path
        self.header = header
        self.values = []

    def add_entry(self, value):
        self.values.append(value)
        if len(self.values) % self.update_frequency == 0:
            mean_value = mean(self.values)

            if self.update_frequency > 1:
                med = median(self.values)
                std = stdev(self.values)
                minimum = min(self.values)
                quart1 = quantile(self.values, 0.25)
                quart3 = quantile(self.values, 0.75)
                maximum = max(self.values)
                scores = [mean_value, std, minimum, quart1, med, quart3, maximum]
            else:
                scores = [mean_value]
            print(self.y_label + ": (min: " + str(min(self.values)) + ", avg: " + str(mean_value) + ", max: " + str(
                max(self.values)))
            print('{"metric": "' + self.y_label + '", "value": {}}}'.format(mean_value))
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
                if self.update_frequency > 1:
                    std.append(float(data[i][1]))
                    minimum.append(float(data[i][2]))
                    quartil1.append(float(data[i][3]))
                    med.append(float(data[i][4]))
                    quartil3.append(float(data[i][5]))
                    maximum.append(float(data[i][6]))

        plt.subplots()
        if self.update_frequency > 1:
            plt.plot(x, med, color='blue')
            plt.fill_between(x, med, quartil1, facecolor='cornflowerblue', interpolate=True)
            plt.fill_between(x, med, quartil3, facecolor='cornflowerblue', interpolate=True)
            plt.fill_between(x, quartil1, minimum, facecolor='lightblue', interpolate=True)
            plt.fill_between(x, quartil3, maximum, facecolor='lightblue', interpolate=True)
        else:
            plt.plot(x, y)

        plt.title(self.header)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # plt.legend(loc="upper left")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def _save_csv(self, path, score):
        if not os.path.exists(path):
            with open(path, "w"):
                pass
        scores_file = open(path, "a", newline='')
        with scores_file:
            writer = csv.writer(scores_file)
            writer.writerow(score)
