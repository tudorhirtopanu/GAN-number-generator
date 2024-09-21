import torch.nn as nn


class Generator(nn.Module):
    """
    A simple Generator model for GANs that takes in a noise vector and outputs
    a flattened image.

    :param z_dim: int
        Dimension of the input noise vector.
    :param img_dim: int
        Dimension of the output image (flattened).
    """
    def __init__(self, z_dim, img_dim):
        """
        Initialises the Generator model.

        :param z_dim: int
            Dimension of the input noise vector.
        :param img_dim: int
            Dimension of the output image (flattened).
        """
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        Forward pass through the generator.

        :param x: torch.Tensor
            Input noise tensor of shape (batch_size, z_dim).
        :return:
            Generated image tensor of shape (batch_size, img_dim).
        :rtype: torch.Tensor
        """
        return self.generator(x)
