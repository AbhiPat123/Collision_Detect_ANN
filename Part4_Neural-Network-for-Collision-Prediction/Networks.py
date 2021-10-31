import torch
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        # initializing the base (super) class of this class - same as super().__init__()
        super(Action_Conditioned_FF, self).__init__()

        self.lin_layer_1 = nn.Linear(6, 125)
        self.lin_layer_2 = nn.Linear(125, 1)

        # non-linear activation to use
        self.nonlinear_activation = nn.Sigmoid()

    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor

        l1_z = self.lin_layer_1(input)
        l1_a = self.nonlinear_activation(l1_z)

        l2_z = self.lin_layer_2(l1_a)
        l2_a = self.nonlinear_activation(l2_z)

        #l3_z = self.lin_layer_3(l2_a)
        #l3_a = self.nonlinear_activation(l3_z)

        output = l2_a
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