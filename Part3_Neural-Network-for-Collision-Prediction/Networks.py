import torch
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    def __init__(self, inp_size=6, l1_size=10, lout_size=1):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        # initializing the base (super) class of this class - same as super().__init__()
        super(Action_Conditioned_FF, self).__init__()
        # Linear Layer 1
        self.lin_layer_1 = nn.Linear(inp_size, l1_size)
        # Linear Layer 2
        self.lin_layer_2 = nn.Linear(l1_size, lout_size)
        # non-linear activation to use
        self.nonlinear_activation = nn.Sigmoid()

    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        # take the network input and pass through hidden layer (no activation yet)
        l1_z = self.lin_layer_1(input)
        # apply nonlinear activation on the previous computation
        l1_a = self.nonlinear_activation(l1_z)
        # pass to the next layer
        output_z = self.lin_layer_2(l1_a)
        # activation at final layer
        output_a = self.nonlinear_activation(output_z)
        output = output_a
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
