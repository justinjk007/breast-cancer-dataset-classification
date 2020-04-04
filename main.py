import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from neural_net import NeuralNetwork
import data


def spit_out_name_from_output(output):
    max_val, _ = torch.max(output, 0)
    index = _.item()
    if (index == 0):  # the index number
        return "malignant"
    else:
        return "benign"

def test_trained_network(ANN):
    print("Trained network being tested...\n")
    error_count = 0

    scaled_input = data.testing_input_mod
    for i in range(len(data.testing_input_mod)):
        _input = torch.tensor(scaled_input[i], dtype=torch.float)
        exp_output = data.testing_output[i]
        print("Expected output  : ", exp_output)
        _output = ANN.predict(_input)
        print("Predicted output : ", spit_out_name_from_output(_output), "\n")
        if (spit_out_name_from_output(_output) != exp_output):
            error_count += 1
    print("Error count after testing", len(data.testing_input), "inputs: ",
          error_count)


def main():
    scaled_input = data.training_input_mod

    # load data into tensor
    _input = torch.tensor(scaled_input, dtype=torch.float)
    _output = torch.tensor(data.training_expected_output, dtype=torch.float)

    # This section is for plotting ##############################
    gene_array = []
    loss_array = []
    fig, ax = plt.subplots()
    ax.set(xlabel='generation',
           ylabel='mean sum squared error',
           title='Neural network, error loss after each generation')
    # This section is for plotting ##############################

    # input,output,hidden layer size
    ANN = NeuralNetwork(i=10, h1=5, h2=2, o=2)
    # weight training
    for i in range(15000):
        # mean sum squared error
        mean_error = torch.mean((_output - ANN(_input))**2).detach().item()
        print("Generation: " + str(i) + " error: " + str(mean_error))
        gene_array.append(i)
        loss_array.append(mean_error)
        ANN.train(_input, _output)

    torch.save(ANN, "algo1.weights")
    # ANN = torch.load("algo1.weights")
    test_trained_network(ANN)
    ax.plot(gene_array, loss_array)
    plt.show()


if __name__ == "__main__":
    main()
