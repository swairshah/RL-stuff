import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FCNet(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):
        x = F.relu(self.fc1(data))
        x = self.fc2(x)
        return x

    def learn(self, data, labels):
        self.optimizer.zero_grad()
        data = torch.tensor(data).to(self.device)
        labels = torch.tensor(labels).to(self.device)
        loss = self.loss(data, labels)
        loss.backward()
        self.optimizer.step()

