import random as rd
import numpy as np


def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [rd.random() for i in range(n_inputs+1)]} for i in range(n_hidden)]
    network.append(hidden_layer)

    output_layer = [{'weights': [rd.random() for i in range(n_hidden+1)]} for i in range(n_outputs)]
    network.append(output_layer)

    return network


def activate(weights, inputs):
    bias = weights[len(weights)-1]
    activation = 0
    for i in range(len(weights)-1):
        activation += (weights[i]*inputs[i])
    activation += bias
    return activation


def transfer(activation):
    output = 1/(1+np.exp(-activation))
    return output


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            neuron['output'] = transfer(activate(neuron['weights'], inputs))
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


def transfer_derivative(output):
    return output*(1.0-output)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


def update_weights(network, row, l_rate):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            for i in range(len(neuron['weights'])-1):
                neuron['weights'][i] += l_rate * neuron['delta'] + inputs[i]
            neuron['weights'][len(neuron['weights'])-1] += l_rate * neuron['delta']
            new_inputs.append(neuron['output'])
        inputs = new_inputs


if __name__ == "__main__":
    rd.seed(1)

    network = initialize_network(3, 2, 2)
    row = [2, 1, 3]
    forward_propagate(network, row)
    backward_propagate_error(network, row)
    print(np.array(network))
    update_weights(network, row, 0.1)
    print(np.array(network))