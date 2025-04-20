# Basic Autoencoder (AE)

This directory contains the implementation of a basic Autoencoder for image compression and reconstruction.

## Directory Structure

- `src/` - Source code for the Autoencoder implementation
- `checkpoints/` - Saved model checkpoints
- `generated_samples/` - Reconstructed images and latent space visualizations
- `experiments/` - Experimental configurations and results

## Model Architecture

The Autoencoder consists of two main components:
- Encoder: Compresses input images into a lower-dimensional latent space
- Decoder: Reconstructs images from the latent space representation

## Training

To train the model:
1. Prepare your dataset
2. Run the training script from the `src/` directory
3. Monitor reconstruction loss and quality

## Results

The model produces:
- Reconstructed images from the training set
- Latent space visualizations
- Compression ratios and reconstruction quality metrics

All outputs are saved in the `generated_samples/` directory.

## Requirements

- Python 3.x
- PyTorch
- Other dependencies as specified in the project's requirements file 