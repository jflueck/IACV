import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN(nn.Module):
    """Convolutional Neural Network.

    Updated architecture with multiple convolutional blocks, batch normalization,
    pooling layers, dropout layers, and fully connected layers as per the suggested design.
    """

    def __init__(self):
        """Initialize layers."""
        super().__init__()
        # Convolutional Block 1
        self.conv1a = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)

        # Convolutional Block 2
        self.conv2a = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)

        # Convolutional Block 3
        self.conv3a = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.25)

        # Calculate the size of the flattened features after the convolutional layers
        self.flattened_size = 128 * 6 * 6  # Adjust if input size changes

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.bnfc1 = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(0.5)

        # Output Layer
        self.fc2 = nn.Linear(512, 10)  # 10 classes for classification

    def forward(self, x):
        """Forward pass of network."""
        # Convolutional Block 1
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Convolutional Block 2
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Convolutional Block 3
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Flatten the output from the convolutional layers
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch

        # Fully Connected Layers
        x = F.relu(self.bnfc1(self.fc1(x)))
        x = self.dropout_fc1(x)

        # Output Layer
        x = self.fc2(x)
        return x

    def write_weights(self, fname):
        """Store learned weights of CNN."""
        torch.save(self.state_dict(), fname)

    def load_weights(self, fname, weights_only=True):
        """Load weights from file in fname."""
        ckpt = torch.load(fname, weights_only=weights_only)
        self.load_state_dict(ckpt)

def get_loss_function():
    """Return the loss function to use during training."""
    return nn.CrossEntropyLoss()

def get_optimizer(network, lr=1e-4, weight_decay=1e-4):
    """Return the optimizer to use during training.

    Uses the Adam optimizer with L2 regularization (weight decay).
    """
    return optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)