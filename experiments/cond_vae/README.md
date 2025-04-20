# Conditional Variational Autoencoder (CVAE)

This directory contains the implementation of a Conditional Variational Autoencoder for controlled image generation.

## Directory Structure

- `src/` - Source code for the CVAE implementation
- `checkpoints/` - Saved model checkpoints
- `generated_samples/` - Generated images and conditional samples

## Model Architecture

The CVAE consists of three main components:
- Encoder: Maps input images and conditions to a latent space distribution
- Latent Space: Represents the learned distribution of the data
- Decoder: Generates images based on both latent vectors and conditions

## Training

To train the model:
1. Prepare your dataset with corresponding conditions
2. Run the training script from the `src/` directory
3. Monitor reconstruction quality and conditional generation

## Results

The model produces:
- Conditionally generated images
- Reconstruction samples
- Latent space interpolations with fixed conditions

All outputs are saved in the `generated_samples/` directory.

## Requirements

- Python 3.x
- PyTorch
- Other dependencies as specified in the project's requirements file 