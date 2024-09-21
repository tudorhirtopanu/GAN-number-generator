import torch.nn as nn


class Discriminator(nn.Module):
    """
    A simple Discriminator model for GANs that takes in a flattened image
    and outputs a probability of the image being real or fake.

    :param img_dim: int
        Dimension of the input image (flattened).
    """
    def __init__(self, img_dim):
        """
        Initialises the Discriminator model.

        :param img_dim: int
            Dimension of the input image (flattened).
        """
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the discriminator.

        :param x: torch.Tensor
            Input image tensor of shape (batch_size, img_dim).
        :return:
            Probability tensor of shape (batch_size, 1), indicating whether the input image is real or fake.
        :rtype: torch.Tensor
        """
        return self.discriminator(x)
