from torch import nn
import torch
import torch.nn.functional as F


class Classify(nn.Module):
    def __init__(self, n, num_classes, hidden_size) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, h_t):
        x = self.fc1(h_t)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


if __name__ == "__main__":

    model_uniform = Classify(5, 10, 2)
    model_uniform.apply(init_weights)

