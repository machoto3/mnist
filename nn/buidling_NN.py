#################################
#      NECESSARY LIBRARIES      #
#################################

import os
import numpy as np
import random
from PIL import Image
import pickle

#################################
#      DEBUG LIBRARIES ETC      #
#################################

import matplotlib.pyplot as plt

#################################
#         SELF IMPORTS          #
#################################

from data_fetch import downloader

#################################

# TODO: - Add GUI (if time permits it) so users can try their own handwrittings - Could be also done as a CLI app
#       - Vectorize processing of mini-batches
#       - Try out different cost/activation functions (So far CCE with respect to Sigmoid is the best
#         might get better results with CCE Softmax. MSE with sigmoids is utter trash, may also try Softmax)
#       - Actually comment the code
#       - Graph out results with different hyperparameters
#       - Write report

#################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derived(x):
    return sigmoid(x) * (1 - sigmoid(x))


def cost_derived(x, y):
    # Cross entropy with sigmoid
    return (x - y) / (x * (1 - x))


class NN(object):
    def __init__(self, sizes=None):
        if sizes is None:
            sizes = [784, 49, 10]

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(b, 1) for b in sizes[1:]]
        self.weights = [np.random.randn(y, w) for w, y in zip(sizes[:-1], sizes[1:])]
        self.logName = None

    def feed_forward(self, data):
        for b, w in zip(self.biases, self.weights):
            data = sigmoid(np.dot(w, data) + b)

        return data

    def back_propagation(self, x, y):
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        current_activation = x
        all_activations = [x]
        activation_vectors = []

        for b, w in zip(self.biases, self.weights):
            current_activation_vector = np.dot(w, current_activation) + b
            activation_vectors.append(current_activation_vector)

            current_activation = sigmoid(current_activation_vector)
            all_activations.append(current_activation)

        delta = cost_derived(all_activations[-1], y) * sigmoid_derived(activation_vectors[-1])  # Cost derivative

        gradient_b[-1] = delta
        gradient_w[-1] = np.dot(delta, all_activations[-2].transpose())

        for layer in range(2, self.num_layers):
            current_activation_vector = all_activations[-layer]
            activation_derived = sigmoid_derived(current_activation_vector)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * activation_derived

            gradient_b[-layer] = delta
            gradient_w[-layer] = np.dot(delta, all_activations[-layer-1].transpose())

        return gradient_b, gradient_w

    def update(self, batch, learning_rate):
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            delta_gradient_b, delta_gradient_w = self.back_propagation(x, y)
            gradient_b = [dgb + gb for dgb, gb in zip(delta_gradient_b, gradient_b)]
            gradient_w = [dgw + gw for dgw, gw in zip(delta_gradient_w, gradient_w)]

        self.biases = [b - (learning_rate / len(batch)) * db for b, db in zip(self.biases, gradient_b)]
        self.weights = [w - (learning_rate / len(batch)) * dw for w, dw in zip(self.weights, gradient_w)]

    def evaluate(self, test_data):
        test_results = [np.argmax(self.feed_forward(x)) == np.argmax(y) for (x, y) in test_data]
        return np.mean(test_results)

    def SGD(self, training_data, epochs, learning_rate, batch_size, test_data):
        if self.logName is None:
            self.logName = "LR" + str(learning_rate).replace('.', '') + "NE" + str(epochs) + "BS" + str(batch_size) + '.txt'

        for epoch in range(epochs):
            random.shuffle(training_data)

            batches = [training_data[k:k + batch_size] for k in range(0, len(training_data), batch_size)]

            learning_rate_current = learning_rate
            if epoch + 1 >= epochs // 2:
                learning_rate_current *= 0.1

            for batch in batches:
                self.update(batch=batch, learning_rate=learning_rate_current)

            with open("accuracy_logs/" + self.logName, 'a') as log:
                log.write("Epoch {} has finished, accuracy is : {}%".format(epoch + 1, self.evaluate(test_data) * 100) + '\n')

            print("Epoch {} has finished, it's accuracy has been logged.".format(epoch + 1))


np.seterr(over='ignore')
network = NN()
data_train, data_test = downloader.load_data()
currently_testing = True  # Change this to try out 0-9 digits that I've written myself

if not os.path.exists('output.nn'):
    data_train, data_test = downloader.load_data()
    var_epochs = 9
    var_learning_rate = 0.500
    var_batch_size = 50

    file_path = "LR" + str(var_learning_rate).replace('.', '') + "NE" + str(var_epochs) + "BS" + str(var_batch_size)

    if os.path.exists("accuracy_logs/" + file_path + ".txt"):
        os.remove("accuracy_logs/" + file_path + ".txt")

    network.SGD(data_test, var_epochs, var_learning_rate, var_batch_size, data_test)

    with open('output.nn', 'wb') as file:
        pickle.dump(network, file)
else:
    with open('output.nn', 'rb') as file:
        network = pickle.load(file)

if currently_testing:
    for num in range(10):
        test = np.array(Image.open("own_tests/" + str(num) + '.jpg').convert("L"))
        print('Actual : {} Guess : {}'.format(num, network.feed_forward(test.reshape(784, 1)).argmax()))

test = np.array(Image.open("som.png").resize((28, 28), resample=Image.LANCZOS).convert("L"))
print(network.feed_forward(test.reshape(784, 1)).argmax())











