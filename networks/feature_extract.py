from torch import nn
import torch
import torch.nn.functional as F


class ObsFeatExtract(nn.Module):
    def __init__(self, frame_size) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.__output_size = 32 * (frame_size // 4) ** 2

    def forward(self, o_t):
        o_t = o_t[:, 0, None, :, :]
        x = self.conv1(o_t)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        return x.flatten(1, -1)

    def output_size(self):
        return self.__output_size


# Spatial state
class PoseFeatExtract(nn.Module):
    def __init__(self, d, n_d) -> None:
        super().__init__()
        self.fc = nn.Linear(d, n_d)

    def forward(self, p_t):
        x = self.fc(p_t)
        return F.relu(x)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


if __name__ == "__main__":
    model_uniform = ObsFeatExtract(5)
