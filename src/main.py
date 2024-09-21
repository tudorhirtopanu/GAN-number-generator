import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.generator import Generator
from src.discriminator import Discriminator

# Set device to GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
learning_rate = 3e-4  # Learning rate for optimizers
z_dim = 64            # Dimension of noise vector (latent space)
image_dim = 28 * 28 * 1   # MNIST images are 28x28x1, flattened to 784
batch_size = 32       # Number of samples per batch
num_epochs = 100      # Number of epochs to train


# Initialise Discriminator and Generator models and move them to device
discriminator = Discriminator(image_dim).to(device)
generator = Generator(z_dim, image_dim).to(device)

# Fixed noise used to visualize generator progress during training
fixed_noise = torch.randn(batch_size, z_dim).to(device)

# Data transformation pipeline for the MNIST dataset
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset and create DataLoader for batching
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Optimisers for the discriminator and generator
opt_discriminator = optim.Adam(discriminator.parameters(), lr=learning_rate)
opt_generator = optim.Adam(generator.parameters(), lr=learning_rate)

# Binary Cross Entropy Loss function used for both discriminator and generator
criterion = nn.BCELoss()

# Set up TensorBoard writers to log real and fake images
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")

step = 0  # Global step for logging

# Training Loop
for epoch in range(num_epochs):

    # Initialise variables for loss tracking
    running_lossD = 0.0
    running_lossG = 0.0
    num_batches = len(loader)

    for batch_idx, (real, _) in enumerate(loader):
        # Flatten real images and move to device (batch_size x 784)
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # -------------------------
        # Train Discriminator
        # -------------------------

        # Generate batch of random noise
        noise = torch.randn(batch_size, z_dim).to(device)

        # Generate fake images from noise
        fake = generator(noise)

        # Discriminator's prediction on real images
        disc_real = discriminator(real).view(-1)

        # Real images label = 1
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))

        # Discriminator's prediction on fake images
        disc_fake = discriminator(fake).view(-1)

        # Fake images label = 0
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        # Average the discriminator's loss
        lossD = (lossD_real + lossD_fake) / 2

        # Zero out gradients before backprop
        discriminator.zero_grad()

        # Backpropagation for discriminator
        lossD.backward(retain_graph=True)

        # Update discriminator weights
        opt_discriminator.step()

        # -------------------------
        # Train Generator
        # -------------------------

        # Discriminator's prediction on fake images
        output = discriminator(fake).view(-1)

        # Generator tries to get label = 1 (real)
        lossG = criterion(output, torch.ones_like(output))

        # Zero out gradients for generator
        generator.zero_grad()

        # Backpropagation for generator
        lossG.backward()

        # Update generator weights
        opt_generator.step()

        # Accumulate losses
        running_lossD += lossD.item()
        running_lossG += lossG.item()

        # Print current batch losses
        print(f"\rEpoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} | Loss D: {lossD:.4f}, Loss G: {lossG:.4f}", end="")

    # Calculate and print average losses for the epoch
    avg_lossD = running_lossD / num_batches
    avg_lossG = running_lossG / num_batches
    print(f"\rEpoch [{epoch}/{num_epochs}] Avg Loss D: {avg_lossD:.4f}, Avg Loss G: {avg_lossG:.4f}", end="")
    print()

    # -------------------------
    # Log generated and real images to TensorBoard
    # -------------------------
    with torch.no_grad():

        # Generate images from fixed noise
        fake = generator(fixed_noise).reshape(-1, 1, 28, 28)

        # Reshape real images for logging
        data = real.reshape(-1, 1, 28, 28)

        # Normalize images for display
        img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
        img_grid_real = torchvision.utils.make_grid(data, normalize=True)

        # Log images to TensorBoard
        writer_fake.add_image("Mnist Fake Images", img_grid_fake, global_step=step)
        writer_real.add_image("Mnist Real Images", img_grid_real, global_step=step)

    # Increment step after each epoch
    step += 1
