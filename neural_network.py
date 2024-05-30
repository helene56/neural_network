import numpy as np
import scipy.special as sp # for the sigmoid function called expit()

class neuralNetwork:
    def __init__(self, inputnodes, hidden_nodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hidden_nodes
        self.onodes = outputnodes
        self.lr = learningrate
        # link weight matrices - setting weights according to normal distribution and standard deviation
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes)) # weigh matrix between input nodes and hidden nodes
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes)) # weigh matrix between output nodes and hidden nodes
        # initilaze the sigmoid function
        self.activation_function = lambda x: sp.expit(x)


    # train the neural network
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # error is  the (target - actual)
        output_error = targets - final_outputs
        # hidden layer error
        hidden_errors = np.dot(self.who.T, output_error) # note: weights transposed dot errors
        # weights updated between hidden and output layers
        self.who += self.lr * np.dot((output_error * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        # weights updated between hidden and input layers (previous layers, same code just the previous layer)
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T # note .T transposes the array, ndmin=2 makes sure that it has at least 2 dimensions
        # signal into the hidden layer nodes
        hidden_inputs = np.dot(self.wih, inputs)
        # signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# array = np.random.rand(3, 3) - 0.5
# print(array)