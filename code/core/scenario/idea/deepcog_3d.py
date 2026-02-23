import torch.nn.functional as F
import torch.nn as nn
import torch

class DeepCog(nn.Module):

    def __init__(self, args):
        """
        PyTorch version of the DeepCog architecture.
        """
        super().__init__()
        H = W = 5 # n row and col
        T = 10 # n look back
        H = 32 # n hidden
        C = 1 # n cluster
        # Conv blocks (channels-first: N, C, D, H, W)
        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=H,
            kernel_size=(3, 3, 3),
            padding=0
        )
        self.conv2 = nn.Conv3d(
            in_channels=H,
            out_channels=H,
            kernel_size=(3, 3, 3),
            padding=0
        )
        self.fc1 = nn.Linear(192, H * 2)
        self.fc2 = nn.Linear(H * 2, H)
        self.out = nn.Linear(H, C)

    def forward(self, x):
        """
        x: Tensor of shape (batch, 1, lookback, H, W)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = torch.flatten(x, start_dim=1)  # same as Keras Flatten()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)  # no activation, matches Keras Dense(num_cluster)
        return x
