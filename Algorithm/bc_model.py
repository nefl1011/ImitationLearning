import tensorflow as tf
import numpy as np
from keras import Sequential
from keras.utils import plot_model
from sklearn import svm

from keras.layers import Dense, Activation, Flatten, Conv2D


class BCModel:
    def __init__(self, env):
        self.save_dir = ""
        self.state_size = env.observation_space.shape[0]
        print(env.observation_space)
        self.action_size = env.action_space.n
        self.hidden_size = 128
        self.env = env
        self.input_shape = (128,)
        self.model = self.create_model()

    def create_model(self):
        print("creating the model");

        model = svm.SVC(gamma='scale', probability=True)

        #  model = Sequential()
        # Adds a densely-connected layer with 128 units to the model:
        #  model.add(Dense(self.state_size, input_shape=self.input_shape, activation='relu'))
        # Add another:
        #  model.add(Dense(self.hidden_size, activation='relu'))
        # Add a softmax layer with 18 output units:
        #  model.add(Dense(self.action_size, activation='softmax'))

        #  model.compile(optimizer='adam',
        #              loss='sparse_categorical_crossentropy',
        #              metrics=['accuracy'])
        return model

    def train_data(self, data, labels, iterations, batch_size):
        print("training data")
        self.model.fit(data, labels)
        print("finished")
        # print(data[0])
        #self.model.fit(data, labels, iterations, batch_size)
        #self.model.evaluate()

    def get_predicted_action_and_probability(self, state):
        s = [np.asarray(state, dtype=np.float32)]
        prediction = self.model.predict(s)
        proba = self.model.predict_proba(s)
        # print("predict action %d with probability %f" %(prediction, np.max(proba[0])))
        return prediction, np.max(proba[0])

    def save_data(self):
        print("save data into directory: " + self.save_dir)
