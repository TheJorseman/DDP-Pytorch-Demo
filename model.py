import torch.nn as nn

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=kwargs["input_shape"], out_features=128),
            nn.ReLU(inplace=True),
            # small dimension
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(inplace=True),
            # Recconstruction of input
            nn.Linear(in_features=128, out_features=kwargs["input_shape"]),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        reconstructed = self.net(features)
        return reconstructed
