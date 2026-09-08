import torch
from torch import nn, Tensor


class SimpleModel(nn.Module):
    def __init__(self):

        super().__init__()
        self.nnet = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 28 * 28, 10),
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