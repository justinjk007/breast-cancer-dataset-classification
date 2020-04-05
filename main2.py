import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
# from neural_net import NeuralNetwork
from neural_net2 import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import data


def spit_out_name_from_output(output):
    max_val, _ = torch.max(output, 0)
    index = _.item()
    if (index == 0):  # the index number
        return "malignant"
    else:
        return "benign"


def test_trained_network(model, test_X, test_Y):
    _input = torch.tensor(test_X, dtype=torch.float)
    out = model(_input)
    error = 0
    tot = 0
    mal_guessed_wrong = 0
    ben_guessed_wrong = 0
    for exp, ans in zip(out, test_Y):
        tot += 1
        max_val, _ = torch.max(exp, 0)
        index = _.item()
        if (index == ans):
            print("match")
        else:
            print("no_match")
            error += 1
            if (index == 0):
                mal_guessed_wrong += 1
            else:
                ben_guessed_wrong += 1

    print("Error is {}".format(error))
    print("Total is {}".format(tot))
    print("Malignant guessed wrong is {}".format(mal_guessed_wrong))
    print("Benign guessed wrong is {}".format(ben_guessed_wrong))


def main():
    # load IRIS dataset
    dataset = pd.read_csv('WDBC_modified.dat')

    # transform species to numerics
    dataset.loc[dataset.diagnosis == 'M', 'diagnosis'] = 0
    dataset.loc[dataset.diagnosis == 'B', 'diagnosis'] = 1

    train_X, test_X, train_Y, test_Y = train_test_split(
        dataset[dataset.columns[2:12]].values,
        dataset.diagnosis.values,
        test_size=0.3,
    )
    train_Y = train_Y.astype(np.float32)
    test_Y = train_Y.astype(np.float32)

    sc = StandardScaler()
    train_X = sc.fit_transform(train_X)
    test_X = sc.fit_transform(test_X)

    # load data into tensor
    _input = torch.tensor(train_X, dtype=torch.float)
    _output = torch.tensor(train_Y, dtype=torch.long)

    # input,output,hidden layer size
    model = NeuralNetwork(i=10, h1=6, h2=2, o=2)

    # loss function and optimizer
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)

    for epoch in range(15000):
        # Forward Pass
        output = model(_input)
        # Loss at each oteration by comparing to target(label)
        loss = lossFunction(output, _output)

        # Backpropogating gradient of loss
        optimizer.zero_grad()
        loss.backward()

        # Updating parameters(weights and bias)
        optimizer.step()
        _loss = loss.item()

        print("Epoch {}, Training loss: {}".format(epoch, _loss / len(_input)))

    torch.save(model, "algo1.weights")
    # model = torch.load("algo1.weights")
    test_trained_network(model, test_X, test_Y)


if __name__ == "__main__":
    main()
