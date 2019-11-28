import tensorflow as tf
import numpy as np
from keras import Sequential
from keras.utils import plot_model

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

        model = Sequential()
        # Adds a densely-connected layer with 128 units to the model:
        model.add(Dense(self.state_size, input_shape=self.input_shape, activation='relu'))
        # Add another:
        model.add(Dense(self.hidden_size, activation='relu'))
        # Add a softmax layer with 18 output units:
        model.add(Dense(self.action_size, activation='softmax'))

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train_data(self, data, labels, iterations, batch_size):
        print("training data")
        # print(data[0])
        self.model.fit(data, labels, iterations, batch_size)

        #self.model.evaluate()

    def save_data(self):
        print("save data into directory: " + self.save_dir)

    def create_model2(self):
        """
        Creates the model.
        """
        state_ph = tf.placeholder(tf.float32, shape=[None, 128])
        # Process the data

        # # Hidden neurons
        with tf.variable_scope("layer1"):
            hidden = tf.layers.dense(state_ph, 128, activation=tf.nn.relu)

        with tf.variable_scope("layer2"):
            hidden = tf.layers.dense(hidden, 128, activation=tf.nn.relu)
        # Make output layers
        with tf.variable_scope("layer3"):
            logits = tf.layers.dense(hidden, 128)
            # Take the action with the highest activation
        with tf.variable_scope("output"):
            action = tf.argmax(input=logits, axis=1)

        return state_ph, action, logits