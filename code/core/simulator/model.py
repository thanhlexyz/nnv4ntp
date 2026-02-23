import torch.nn.functional as F
import torch.nn as nn
import torch

def create(args):
    return DeepCog(args).to(args.device)

class DeepCog(nn.Module):
    def __init__(self, args):
        """
        PyTorch version of the DeepCog architecture using only Conv2d layers.
        Processes each time step independently through spatial convs (using groups),
        then applies temporal convs.

        Assumes input of shape: (batch, T, H, W)
        """
        super().__init__()
        # Model width / hidden channels
        hidden = 32
        C = 1        # number of clusters (output classes)
        T = 10       # number of time steps (input channels)

        # --- Spatial convs (per time step): grouped Conv2d ---
        # Process each time step independently using groups=T
        # Each time slice goes through spatial convs independently
        self.spatial_conv1 = nn.Conv2d(
            in_channels=T,
            out_channels=T * hidden,
            kernel_size=3,
            padding=0,
            groups=T
        )
        self.spatial_conv2 = nn.Conv2d(
            in_channels=T * hidden,
            out_channels=T * hidden,
            kernel_size=3,
            padding=0,
            groups=T
        )

        # --- Temporal convs: Conv2d with temporal kernel ---
        # Mix across channels (time steps), using spatial dims as temporal
        # After spatial convs, spatial dims are reduced to 1x1, so we use 1x1 spatial kernel
        # Temporal mixing: reduce T*hidden -> hidden channels
        self.temporal_conv1 = nn.Conv2d(
            in_channels=T * hidden,
            out_channels=hidden * 8,  # T-2 equivalent in channel reduction
            kernel_size=1,
            padding=0
        )
        self.temporal_conv2 = nn.Conv2d(
            in_channels=hidden * 8,
            out_channels=hidden * 6,  # T-4 equivalent
            kernel_size=1,
            padding=0
        )

        # --- Fully connected head ---
        self.fc1 = nn.Linear(hidden * 6, hidden * 2)
        self.fc2 = nn.Linear(hidden * 2, hidden)
        self.out = nn.Linear(hidden, C)

    def forward(self, x):
        """
        x: Tensor of shape (batch, T, H, W)
        """
        # Spatial convs applied independently per time step (via groups)
        x = F.relu(self.spatial_conv1(x))     # (batch, T*hidden, H', W')
        x = F.relu(self.spatial_conv2(x))     # (batch, T*hidden, H'', W'')

        # Temporal convs mixing across time steps
        x = F.relu(self.temporal_conv1(x))    # (batch, hidden*8, H'', W'')
        x = F.relu(self.temporal_conv2(x))    # (batch, hidden*6, H'', W'')

        # Global average pooling over spatial dimensions
        x = F.adaptive_avg_pool2d(x, (1, 1))  # (batch, hidden*6, 1, 1)
        x = torch.flatten(x, start_dim=1)     # (batch, hidden*6)

        # Fully connected head
        x = F.relu(self.fc1(x))               # (batch, hidden*2)
        x = F.relu(self.fc2(x))               # (batch, hidden)
        x = self.out(x)                       # (batch, C)
        return x

if __name__ == "__main__":
    model = DeepCog(None)
    model.eval()
    x = torch.rand(10, 10, 5, 5)
    print(model)
    print(model(x).shape)

    torch.onnx.export(
        model,
        x,
        'test.onnx',
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    )
