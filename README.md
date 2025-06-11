# Generative Models for Handwritten Digit Generation

This project implements and compares various generative models for creating handwritten digits using the MNIST dataset. The project includes implementations of multiple architectures, from basic autoencoders to advanced generative models.

## Implemented Models

- Basic Autoencoder (AE)
- Convolutional Variational Autoencoder (Conv-VAE)
- Conditional Convolutional VAE
- Vector Quantized VAE (VQ-VAE)
- Generative Adversarial Network (GAN)
- Diffusion Model

Each model is implemented as a separate experiment in the `experiments/` directory.

## Features

- Multiple generative model architectures
- Support for CPU, CUDA, and Apple Metal (MPS) devices
- Training visualization with loss plots and reconstruction samples
- Generation of new handwritten digits
- Checkpoint system with best model tracking
- Comparative analysis of different architectures

## Requirements

- Python 3.8+
- uv (Python package installer)
- PyTorch 2.2.0+ (installation method depends on your system)
- Other dependencies listed in `requirements.txt`

## Installation

1. Install uv if you haven't already:
```bash
pip install uv
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies using uv:
```bash
uv pip install -r requirements.txt
```

Note: The `requirements.txt` file is configured for Apple Silicon Macs by default. If you're using a different system:
- For NVIDIA GPU (CUDA): Comment out the Mac-specific PyTorch lines and uncomment the standard PyTorch installation lines
- For CPU-only: Use the standard PyTorch installation lines

## Project Structure

```
.
├── README.md
├── requirements.txt
├── experiments/
│   ├── basic_ae/      # Basic Autoencoder implementation
│   ├── conv_vae/      # Convolutional VAE implementation
│   ├── cond_vae/      # Conditional Convolutional VAE
│   ├── vq_vae/        # Vector Quantized VAE
│   ├── gan/           # Generative Adversarial Network
│   ├── diffusion/     # Diffusion Model
│   └── data/          # Shared data directory
├── presentation_1/    # First presentation materials
└── presentation_2/    # Second presentation materials
```

## Running Experiments

Each model has its own directory in `experiments/` with specific training and generation scripts. Navigate to the desired model's directory and follow its README for specific instructions.

### Common Parameters

Most models support these common parameters:
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate
- `--latent_dim`: Dimension of the latent space (where applicable)

### Example: Training a Model

```bash
cd experiments/conv_vae
python train.py --epochs 500 --latent_dim 2
```

### Example: Generating Digits

```bash
cd experiments/conv_vae
python generate.py --num_samples 10
```

## Results and Analysis

The project includes two presentations that analyze and compare the results of different models:
- `presentation_1/`: Initial results and model comparisons
- `presentation_2/`: Final analysis and conclusions

## Device Support

All models automatically select the best available device in the following order:
1. NVIDIA CUDA GPU (if available)
2. Apple Metal (MPS) for Apple Silicon/Intel Macs
3. CPU (fallback)

No additional configuration is needed - the code will automatically detect and use the best available device.

## License

This project is licensed under the terms of the included LICENSE file.