from torch import nn
import torch.nn.functional as F
import torch

# Inter-Agent Communication


# Message generatoion
class MessageGenNet(nn.Module):
    def __init__(self, n, n_m, hidden_size):
        super().__init__()

        self.fc1 = nn.Linear(n, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_m)

    def forward(self, h_t):
        x = self.fc1(h_t)
        x = F.relu(x)
        return self.fc2(x)


# Message decode
class MessageDecodeNet(nn.Module):
    def __init__(self, n_m, n):
        super().__init__()

        self.fc = nn.Linear(n_m, n)
        pass

    def forward(self, m_t):
        x = self.fc(m_t)
        return F.relu(x)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


if __name__ == "__main__":

    model_uniform = MessageGenNet(5, 10, 2)
    model_uniform.apply(init_weights)

    model_uniform = MessageDecodeNet(5, 10)
    model_uniform.apply(init_weights)
