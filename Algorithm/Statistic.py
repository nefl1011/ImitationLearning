from statistics import mean
import os
import csv
import matplotlib

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
            print(self.y_label + ": (min: " + str(min(self.values)) + ", avg: " + str(mean_value) + ", max: " + str(
                max(self.values)))
            print('{"metric": "' + self.y_label + '", "value": {}}}'.format(mean_value))
            self._save_csv(self.directory_path + self.y_label + ".csv", mean_value)
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
        with open(input_path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            for i in range(0, len(data)):
                x.append(float(i) * small_batch_length)
                y.append(float(data[i][0]))

        plt.subplots()
        plt.plot(x, y, label="last " + str(small_batch_length) + " average")

        batch_averages_y = []
        batch_averages_x = []
        temp_values_in_batch = []
        relative_batch_length = big_batch_length / small_batch_length

        for i in range(len(y)):
            temp_values_in_batch.append(y[i])
            if (i + 1) % relative_batch_length == 0:
                if not batch_averages_y:
                    batch_averages_y.append(mean(temp_values_in_batch))
                    batch_averages_x.append(0)
                batch_averages_x.append(len(batch_averages_y) * big_batch_length)
                batch_averages_y.append(mean(temp_values_in_batch))
                temp_values_in_batch = []
        if len(batch_averages_x) > 1:
            plt.plot(batch_averages_x, batch_averages_y, linestyle="--",
                     label="last " + str(big_batch_length) + " average")

        plt.title(self.header)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc="upper left")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def _save_csv(self, path, score):
        if not os.path.exists(path):
            with open(path, "w"):
                pass
        scores_file = open(path, "a")
        with scores_file:
            writer = csv.writer(scores_file)
            writer.writerow([score])
