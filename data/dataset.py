from os.path import abspath, dirname, join

from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch


def image_loader(path):
    return Image.open(open(path, "rb")).convert("RGB")


class Normalize:
    def __call__(self, x: torch.tensor) -> torch.tensor:
        return (x - x.mean()) / x.std()


class MNIST(ImageFolder):
    def __init__(self, transform) -> None:

        super().__init__(
            abspath(join(dirname(abspath(__file__)), "all_png")),  # root path
            transform,
            loader=image_loader,
        )


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), Normalize()])
    data = MNIST(transform)
