import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend

import matplotlib.pyplot as plt
from neural_network import neuralNetwork

# number of nodes
input_nodes = 784 # 785 numbers that contain info about a number. the first number is the label so to display the number: 28x28 = 784
hidden_nodes = 100 # 100 lines in total in data, just chosen
output_nodes = 10 # want it to output an integer from 0-9

# learning rate
learning_rate = 0.3

n = neuralNetwork(input_nodes, output_nodes, hidden_nodes, learning_rate)

inputs_list = [1.0, 0.5, -1.5]

# print(n.query(inputs_list))

with open("mnist_handwritten_dataset/sub_dataset/mnist_train_100.csv", 'r') as training_data_file:
    training_data_list = training_data_file.readlines()

print(len(data_list))


all_values = data_list[1].split(',') # split the first data in the list at every comma, this splits the data for the second number
print(len(all_values))
image_array = np.asfarray(all_values[1:]).reshape((28,28)) # slice to start from the second data (1:), then numpy.asfarray turns strings to numbers and create array, reshape rearranges array into a desired matrix (28 x 28)
plt.imshow(image_array, cmap='Greys', interpolation='None') # imshows plot a rectangular array

# Save the plot to a file
# plt.savefig('plot.png')

scaled_input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 # scaled input to go from 0.01 - 1.00. note: only output should avoid 1.00 as it is 'impossible' for it to reach it
print(scaled_input)

# note: what should the output be? preferably a number that is within the range 0-9 as we want it to label an image of a number

# output node is 10 (example)
onodes = 10
targets = np.zeros(onodes) + 0.01 # setting up an array with 0 then adding 0.01 to all of the zero
targets[int(all_values[0])] = 0.99
print(targets)

# setting up a loop to handle all the records in the training set
for record in training_data_list:
    