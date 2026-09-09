import torch
from torch import nn, Tensor


class SimpleModel(nn.Module):
    def __init__(self):

        super().__init__()
        self.nnet = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x: Tensor):
        return self.nnet(x)


def get_model() -> nn.Module:
    return SimpleModel()


def main():
    nnet: nn.Module = get_model()
    torch.save(nnet.state_dict(), "dummy.pt")


if __name__ == "__main__":
    main()