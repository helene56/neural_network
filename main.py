from neural_network import neuralNetwork

# number of nodes
input_nodes = 3
hidden_nodes = 3
output_nodes = 3

# learning rate
learning_rate = 0.3

n = neuralNetwork(input_nodes, output_nodes, hidden_nodes, learning_rate)

inputs_list = [1.0, 0.5, -1.5]

print(n.query(inputs_list))
