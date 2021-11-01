from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle


def train_model(no_epochs):

    batch_size = 32
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()

    # set the learning rate optimizer and loss function
    learning_rate = 0.001

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    loss_function = nn.BCELoss()

    tr_losses_to_plot = []
    ts_losses_to_plot = []

    for epoch_i in range(no_epochs):
        model.train()

        # collect the losses for each mini-batch
        tr_mbatch_loss = []

        for idx, sample in enumerate(data_loaders.train_loader): # sample['input'] and sample['label']
            mbatch_output = model(sample['input'])

            mbatch_loss = loss_function(torch.flatten(mbatch_output), sample['label'])

            tr_mbatch_loss.append(mbatch_loss.item())

            # zero out all parameters' gradient values - before performing every step of backward
            optimizer.zero_grad()

            # run one step of the BP Algorthm
            # this accummulates the gradients value of each parameter in the parameters' .grad field
            mbatch_loss.backward()

            # run one step of optimizer (one step of Gradient Descent) - actually updates the parameters
            optimizer.step()

        # add the average of the train losses over the mini-batches
        epoch_tr_loss = sum(tr_mbatch_loss)/len(tr_mbatch_loss)
        tr_losses_to_plot.append(epoch_tr_loss)

        epoch_ts_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
        ts_losses_to_plot.append(epoch_ts_loss)

    # plot the training, testing losses against epochs
    epoch_range = range(0,no_epochs)
    fig, ax = plt.subplots()
    ax.plot(epoch_range, tr_losses_to_plot)
    ax.plot(epoch_range, ts_losses_to_plot)

    ax.set(xlabel='Number of Epochs', ylabel='Loss values',
           title='TRAIN/TEST LOSS versus EPOCHS')
    ax.grid()

    fig.savefig("loss_train_test.png")
    #plt.show()

    # save the model as pickle in saved folder
    torch.save(model.state_dict(), "saved/saved_model.pkl", _use_new_zipfile_serialization=False)

if __name__ == '__main__':
    no_epochs = 500
    train_model(no_epochs)
