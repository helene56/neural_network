import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend

import matplotlib.pyplot as plt
from neural_network import neuralNetwork

# Load the trained model
with open('trained_model_full.pkl', 'rb') as f:
    n = pickle.load(f)

# testing the network
with open("mnist_handwritten_dataset/full_dataset/mnist_test.csv", 'r') as test_data_file:
    test_data_file = test_data_file.readlines()
# # get the first test record to test manually
# all_values = test_data_file[0].split(',')
# print(f"test number: {all_values[0]}")
# image_array = np.asfarray(all_values[1:]).reshape((28,28))
# plt.imshow(image_array, cmap='Greys', interpolation='None')
# # Save the plot to a file
# plt.savefig('plot.png')
# # query the network
# input_list = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
# final_outputs = n.query(input_list)
# print(final_outputs)

# testing the neural network and keeping score
scorecard = []
for record in test_data_file:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    input_list = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    final_outputs = n.query(input_list)
    network_label = np.argmax(final_outputs)
    # print(f"correct label: {correct_label}")
    # print(f"network label: {network_label}")
    if network_label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)

scorecard_array = np.asarray(scorecard)
performance = scorecard_array.sum() / scorecard_array.size
print(f"performance of network: {performance}")

# # example plot data
# all_values = training_data_list[1].split(',') # split the first data in the list at every comma, this splits the data for the second number
# print(len(all_values))
# image_array = np.asfarray(all_values[1:]).reshape((28,28)) # slice to start from the second data (1:), then numpy.asfarray turns strings to numbers and create array, reshape rearranges array into a desired matrix (28 x 28)
# plt.imshow(image_array, cmap='Greys', interpolation='None') # imshows plot a rectangular array

# # Save the plot to a file
# # plt.savefig('plot.png')

# scaled_input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 # scaled input to go from 0.01 - 1.00. note: only output should avoid 1.00 as it is 'impossible' for it to reach it
# print(scaled_input)

# # note: what should the output be? preferably a number that is within the range 0-9 as we want it to label an image of a number

# # output node is 10 (example)
# onodes = 10
# targets = np.zeros(onodes) + 0.01 # setting up an array with 0 then adding 0.01 to all of the zero
# targets[int(all_values[0])] = 0.99
# print(targets)