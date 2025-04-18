# Handwritten Digit Generation using Convolutional VAE

This project implements a convolutional variational autoencoder (VAE) for generating handwritten digits using the MNIST dataset. The model uses a modern architecture with convolutional layers for better image processing capabilities. The project supports training and inference on multiple devices including CPU, NVIDIA CUDA GPUs, and Apple Metal (MPS).

## Features

- Convolutional Variational Autoencoder implementation
- 2D latent space for better visualization and interpolation
- Convolutional layers with batch normalization and dropout
- Support for CPU, CUDA, and Apple Metal (MPS) devices
- Training visualization with loss plots and reconstruction samples
- Generation of new handwritten digits from random latent vectors
- Checkpoint system with best model tracking

## Requirements

- Python 3.8+
- PyTorch 2.2.0+ (installation method depends on your system)
- Other dependencies listed in `requirements.txt`

## Installation

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Before installing requirements, check the `requirements.txt` file:
   - For Apple Silicon Macs: Use as is (default configuration)
   - For other systems (CPU/CUDA): Comment out the Mac-specific PyTorch installation lines and uncomment the standard PyTorch installation lines

3. Install all requirements:
```bash
pip install -r requirements.txt
```

Note: If you're using NVIDIA GPU (CUDA) and want to ensure CUDA support, you can manually install PyTorch first:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Model Architecture

The VAE uses a modern convolutional architecture:

### Encoder
- Input: 28x28 grayscale images
- Conv2D (32 filters, 3x3, stride 2) → BatchNorm → LeakyReLU → (14x14)
- Conv2D (64 filters, 3x3, stride 2) → BatchNorm → LeakyReLU → (7x7)
- Dropout(0.25)
- Flatten → Linear layers for mu and log_var (2D latent space)

### Decoder
- Linear → Reshape to (64, 7, 7)
- Dropout(0.25)
- ConvTranspose2D (32 filters) → BatchNorm → LeakyReLU → (14x14)
- ConvTranspose2D (1 filter) → Sigmoid → (28x28)

## Usage

### Training

To train the model, use the `train.py` script:

```bash
python src/train.py --epochs 500 --latent_dim 2
```

Parameters:
- `--latent_dim`: Dimension of the latent space (default: 2)
- `--epochs`: Number of training epochs (default: 500)
- `--batch_size`: Batch size for training (default: 128)
- `--learning_rate`: Learning rate (default: 1e-4)

During training, the following will be created:
- A timestamped directory under `checkpoints/run_TIMESTAMP/` for each training run
- Checkpoints saved every 10 epochs in the run directory
- Best model saved in `checkpoints/best_model/`
- Training visualizations saved in the run directory

### Generating Digits

After training, you can generate new digits using the `generate.py` script:

```bash
python src/generate.py --num_samples 10
```

Parameters:
- `--latent_dim`: Dimension of the latent space (must match training, default: 2)
- `--num_samples`: Number of digits to generate (default: 10)

## Project Structure

```
.
├── README.md
├── requirements.txt
├── checkpoints/
│   ├── best_model/    # Stores the best model
│   └── run_TIMESTAMP/ # Created for each training run
└── src/
    ├── model.py       # VAE model definition
    ├── train.py       # Training script
    └── generate.py    # Generation script
```

## Output Files

Each training run creates a timestamped directory containing:
- Checkpoint files saved every 10 epochs
- Training loss plot
- Sample reconstructions during training
- Final model state

The best model is maintained in:
- `checkpoints/best_model/vae_best.pth`

Generated digits are saved with timestamps in the root directory.

## Device Support

The code automatically selects the best available device in the following order:
1. NVIDIA CUDA GPU (if available)
2. Apple Metal (MPS) for Apple Silicon/Intel Macs
3. CPU (fallback)

No additional configuration is needed - the code will automatically detect and use the best available device.