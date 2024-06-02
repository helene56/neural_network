import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend

import matplotlib.pyplot as plt
from neural_network import neuralNetwork

# number of nodes
input_nodes = 784 # 785 numbers that contain info about a number. the first number is the label so to display the number: 28x28 = 784
hidden_nodes = 100 # 100 lines in total in data, just chosen, not a perfect number, just experimentate
output_nodes = 10 # want it to output an integer from 0-9

# learning rate
learning_rate = 0.3

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

with open("mnist_handwritten_dataset/sub_dataset/mnist_train_100.csv", 'r') as training_data_file:
    training_data_list = training_data_file.readlines()

# setting up a loop to handle all the records in the training set and train the neural network
for record in training_data_list:
    all_values = record.split(',')
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = np.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99 # the first value in the record is the number label
    # train the network
    n.train(inputs, targets)

# save the trained model
with open("trained_model.pkl", 'wb') as f:
    pickle.dump(n, f)