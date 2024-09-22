# GAN Number Generator

This project implements a Generative Adversarial Network to generate images of handwritten digits based on the MNIST dataset.

## Table of Contents

- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Train the GAN](#1-train-the-gan)
  - [Visualize Training with TensorBoard](#2-visualize-training-with-tensorboard)
- [Results](#results)


## How It Works

1. **Generator**: The generator network takes a random noise vector (latent space) as input and generates 28x28 grayscale images (similar to MNIST digits).
   
2. **Discriminator**: The discriminator network takes either real images from the MNIST dataset or generated images from the generator, and outputs a probability indicating whether the image is real or fake.

3. **Training Process**: The generator and discriminator are trained in a competitive process:
   - The **discriminator** learns to differentiate between real MNIST images and fake images.
   - The **generator** tries to fool the discriminator by generating images that resemble real MNIST digits.
   
   During training, the generator improves its ability to create realistic images, while the discriminator gets better at detecting fake images.

## Requirements

- PyTorch
- TorchVision
- TensorBoard

You can install the dependencies by running:

```
pip install -r requirements.txt
```

## Usage

### 1. Train the GAN

To train the GAN, simply run main.py

The training process will automatically download the MNIST dataset (if not already available), initialize the generator and discriminator, and begin training. The logs for training progress will be saved for visualization.

### 2. Visualize Training with TensorBoard

To monitor the training process, including the generator and discriminator losses and the generated images:

1. Start the TensorBoard server by running the following command in the project directory:

```
tensorboard --logdir=src/logs
```

2. Open your web browser and go to:

```
http://localhost:6006
```

Here, you can visualize the losses over time and see how the generated images improve as training progresses.

## Results

As training progresses, the generator will learn to create increasingly realistic MNIST-like digits. You can monitor the quality of generated images using TensorBoard.

