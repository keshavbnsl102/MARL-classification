from torch import nn
import torch
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, inp_dim, n) -> None:
        super().__init__()
        self.lstm = nn.LSTMCell(inp_dim, n)

    def forward(self, h_t, c_t, u_t):

        n_agents, batch_size, _ = h_t.size()

        h_t = h_t.flatten(end_dim=1)
        c_t = c_t.flatten(end_dim=1)
        u_t = u_t.flatten(end_dim=1)

        h_t_plus_1, c_t_plus_1 = self.lstm(u_t, (h_t, c_t))
        return (
            h_t_plus_1.view(n_agents, batch_size, -1),
            c_t_plus_1.view(n_agents, batch_size, -1),
        )


if __name__ == "__main__":

    model_uniform = LSTM(5, 10)

