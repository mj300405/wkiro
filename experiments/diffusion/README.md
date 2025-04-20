# Diffusion Model

This directory contains the implementation of a diffusion model for image generation. The model follows the Denoising Diffusion Probabilistic Models (DDPM) approach.

## Directory Structure

- `src/`: Contains the model implementation and training code
- `data/`: Directory for storing and processing training data
- `checkpoints/`: Saved model checkpoints
- `generated_samples/`: Generated images during training and inference

## Model Architecture

The diffusion model consists of:
- A U-Net backbone for the denoising process
- A noise scheduler for the forward and reverse diffusion processes
- Training and sampling procedures

## Usage

1. Prepare your dataset in the `data/` directory
2. Train the model using `train.py`
3. Generate samples using `generate.py`

## Requirements

See the main project's `requirements.txt` for dependencies.

## Training

To train the model:
```bash
python src/train.py
```

## Generation

To generate samples:
```bash
python src/generate.py
``` 