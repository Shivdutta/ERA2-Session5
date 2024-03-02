
"""model architectures"""
# Necessary Imports
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    This class defines the structure of the neural network.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer with 1 input channel and 32 output channels.
        conv2 (nn.Conv2d): Second convolutional layer with 32 input channels and 64 output channels.
        conv3 (nn.Conv2d): Third convolutional layer with 64 input channels and 128 output channels.
        conv4 (nn.Conv2d): Fourth convolutional layer with 128 input channels and 256 output channels.
        fc1 (nn.Linear): First fully connected layer with 4096 input features and 50 output features.
        fc2 (nn.Linear): Second fully connected layer with 50 input features and 10 output features.

    Methods:
        __init__(): Constructor method to initialize the neural network layers.
        forward(x): Forward pass for model training.

    """
    def __init__(self):
        """
        Constructor method to initialize the neural network layers.
        """
        super(Net, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)

        # Fully Connected Layers
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        Forward pass for model training.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor of the model.
        """
        x = F.relu(self.conv1(x), 2)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x), 2)
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
