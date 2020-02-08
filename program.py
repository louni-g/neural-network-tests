import random as rd
import numpy as np
import csv


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
        newInputs = []
        for neuron in layer:
            newWeights = []
            for i in range(len(neuron['weights']) - 1):
                newWeights.append(neuron['weights'][i] + l_rate * neuron['delta'] * inputs[i])
            newInputs.append(neuron['output'])
            newWeights.append((neuron['weights'][-1] + l_rate * neuron['delta']))
            neuron['weights'] = newWeights
        inputs = newInputs


def train_network(network, train, l_rate, n_epoch, n_outputs):
    # n_epoch defines number of iterations for the training process
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row[:-1])
            expected = [0 for i in range(n_outputs)]
            expected[int(row[-1])] = 1
            for i in range(n_outputs):
                sum_error += np.sqrt((expected[i] - outputs[i])**2)
            backward_propagate_error(network, expected)  # call the backpropagation function for network and expected vector for computing and adding delta to the network
            update_weights(network, row[:-1], l_rate)  # call update_weights for modifying weights in the network
        print('> epoch = %d, lrate = %.3f, error = %.3f' % (epoch, l_rate, sum_error))


def predict(network, row):
    # network = trained network
    probability = forward_propagate(network, row)
    print(probability)
    return np.argmax(probability)


def load_csv(filename):
    dataset=[]
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t') # does not parse ?
        for row in reader:
            row_float = []
            row_split = row[0].split('\t')
            for entry in row_split:
                row_float.append(float(entry))
            dataset.append(row_float)

    # normalization :
    for i in range(len(dataset)):
        for j in range(len(dataset[i])-1):
          dataset[i][j] = (dataset[i][j] - np.mean([row[j] for row in dataset]))/np.std([row[j] for row in dataset])
    return dataset


def split_dataset(dataset, p):
    rd.shuffle(dataset)
    train_data = []
    test_data = []
    n = int(p*len(dataset))
    for i in range(n):
        train_data.append(dataset[i])
    for i in range(n, len(dataset)):
        test_data.append(dataset[i])
    return train_data, test_data


if __name__ == "__main__":
    rd.seed()

    network = initialize_network(7, 16, 3)
    dataset = load_csv("seeds_dataset.csv")
    #print(np.array(dataset))
    for row in dataset:
        row[-1]=row[-1]-1 # outputs were 1,2 or 3 and are now 0,1 or 2 for practical reasons 

    train_data, test_data = split_dataset(dataset, 0.8)


    train_network(network, train_data, 0.5, 800, 3)
    success_rate = 0
    for test_row in test_data:
        prediction = predict(network, test_row[:-1])
        print(test_row[-1], prediction)
        if test_row[-1] == prediction: success_rate+=1
    print("success rate: ", success_rate*100/len(test_data), "%")





