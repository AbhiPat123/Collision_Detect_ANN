import torch
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    def __init__(self, *args):#inp_size=6, l1_size=10, lout_size=1):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        # initializing the base (super) class of this class - same as super().__init__()
        super(Action_Conditioned_FF, self).__init__()

        # the following variables stores all the layers
        self.lin_layers = nn.ModuleList([])

        # non-linear activation to use
        self.nonlinear_activation = nn.Sigmoid()

        # we need a default case where args is empty (the layer sizes would be 6,10,1 by default)
        if len(args) == 0:
            lay_sizes = (6,10,1)
        else:
            lay_sizes = args

        # create and append a layer for the different sizes
        for l_size in range(len(lay_sizes)-1):
            # get the current and next layer size
            cur_l_size = lay_sizes[l_size]
            nex_l_size = lay_sizes[l_size+1]

            # linear layer for arg_val
            self.lin_layers.append(nn.Linear(cur_l_size, nex_l_size))        

    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        # start with input value which goes through multiple layers and changes values
        val_changes = input

        # go through the self.lin_layers
        for layer in self.lin_layers:
            # get output of layer
            val_changes = layer(val_changes)
            # send through activation
            val_changes = self.nonlinear_activation(val_changes)

        # the above for loop ends with the final output
        output = val_changes

        # return the network output value
        return output

    def evaluate(self, model, test_loader, loss_function):
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.
        # create torch tensors to store all outputs and targets
        model_outputs = torch.empty(0)
        target_labels = torch.empty(0)

        # enumerate through each sample in test_loader
        for idx, sample in enumerate(test_loader):
            # get the sample input and label
            s_input = sample['input']
            s_label = sample['label']

            # forward pass through the model
            s_output = model(s_input)

            # collect the outputs in model_outputs
            model_outputs = torch.cat((model_outputs, s_output))
            # collect the labels in target_labels
            target_labels = torch.cat((target_labels, s_label))

        # computing the loss between computed outputs and targets values
        # need to flatten the tensor for model outputs AND also give the item value
        loss = loss_function(torch.flatten(model_outputs), target_labels).item()
        return loss

def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()
