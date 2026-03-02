import torch
import torch.nn as nn

class NeuralMarkovNet(nn.Module):
    def __init__(self, num_states=4):
        super(NeuralMarkovNet, self).__init__()
        # Input: one-hot state (4) + normalized age (1) = 5 features
        self.fc1 = nn.Linear(num_states + 1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_states)
        
        # We removed self.softmax because it is integrated into nn.CrossEntropyLoss

    def forward(self, state_oh, norm_age):
        # Concatenate current state and age for the 23,538 transitions
        x = torch.cat((state_oh, norm_age), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # Return raw logits to ensure stable gradients during training
        return self.fc3(x)