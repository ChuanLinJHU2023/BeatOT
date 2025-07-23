import torch
import torch.nn as nn
import torch.optim as optim

class SimpleClassifier(nn.Module):
    def __init__(self, list_of_num_hidden_units, input_size = 2):
        super(SimpleClassifier, self).__init__()
        # Build hidden layers dynamically based on the list
        layers = []
        for hidden_units in list_of_num_hidden_units:
            layers.append(nn.Linear(input_size, hidden_units))
            layers.append(nn.ReLU())
            input_size = hidden_units
        # Final output layer
        self.hidden_layers = nn.Sequential(*layers)
        self.output = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = torch.sigmoid(self.output(x))
        return x