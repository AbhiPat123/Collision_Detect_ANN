from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle


def train_model(no_epochs):

    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()

    # set the learning rate optimizer and loss function
    learning_rate = 0.001

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    loss_function = nn.MSELoss()

    losses = []
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(min_loss)

    for epoch_i in range(no_epochs):
        model.train()
        for idx, sample in enumerate(data_loaders.train_loader): # sample['input'] and sample['label']
            # zero out all parameters' gradient values - before performing every step of backward
            optimizer.zero_grad()

            sample_output = model(sample['input'])

            sample_loss = loss_function(torch.flatten(sample_output), sample['label'])

            # run one step of the BP Algorthm
            # this accummulates the gradients value of each parameter in the parameters' .grad field
            sample_loss.backward()

            # run one step of optimizer (one step of Gradient Descent) - actually updates the parameters
            optimizer.step()

    # save the model as pickle in saved folder
    torch.save(model.state_dict(), "saved/saved_model.pkl", _use_new_zipfile_serialization=False)

if __name__ == '__main__':
    no_epochs = 250
    train_model(no_epochs)