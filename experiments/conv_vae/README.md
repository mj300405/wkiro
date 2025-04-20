# Convolutional Variational Autoencoder (Conv-VAE)

This directory contains the implementation of a Convolutional Variational Autoencoder for image generation and compression.

## Directory Structure

- `src/` - Source code for the Conv-VAE implementation
- `checkpoints/` - Saved model checkpoints
- `generated_samples/` - Generated images and reconstructions
- `experiments/` - Experimental configurations and results

## Model Architecture

The Conv-VAE consists of three main components:
- Convolutional Encoder: Processes images through convolutional layers to create latent representations
- Latent Space: Represents the learned distribution of the data
- Convolutional Decoder: Generates images from latent vectors using transposed convolutions

## Training

To train the model:
1. Prepare your dataset
2. Run the training script from the `src/` directory
3. Monitor reconstruction quality and latent space organization

## Results

The model produces:
- Generated images from the latent space
- Reconstruction samples
- Latent space visualizations and interpolations

All outputs are saved in the `generated_samples/` directory.

## Requirements

- Python 3.x
- PyTorch
- Other dependencies as specified in the project's requirements file 