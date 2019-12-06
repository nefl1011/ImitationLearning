import cv2
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

        # A buffer that keeps the last 3 images
        self.process_buffer = []
        # Initialize buffer with the first frame
        #s1, r1, _, _ = self.env.step(0)
        #s2, r2, _, _ = self.env.step(0)
        #s3, r3, _, _ = self.env.step(0)
        #self.process_buffer = [s1, s2, s3]

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

    def convert_process_buffer(self):
        """Converts the list of NUM_FRAMES images in the process buffer
        into one training sample"""
        black_buffer = [cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (84, 90)) for x in self.process_buffer]
        black_buffer = [x[1:85, :, np.newaxis] for x in black_buffer]
        return np.concatenate(black_buffer, axis=2)
