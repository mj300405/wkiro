# Handwritten Digit Generation using Autoencoders

This project implements both standard and variational autoencoders (VAE) for generating handwritten digits using the MNIST dataset. The project supports training and inference on multiple devices including CPU, NVIDIA CUDA GPUs, and Apple Metal (MPS).

## Features

- Standard Autoencoder implementation
- Variational Autoencoder implementation
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

## Usage

### Training

To train the model, use the `train.py` script. You can choose between standard and variational autoencoders:

```bash
# Train standard autoencoder
python src/train.py --model_type standard --epochs 500 --latent_dim 32

# Train variational autoencoder
python src/train.py --model_type variational --epochs 500 --latent_dim 32
```

Parameters:
- `--model_type`: Type of autoencoder ('standard' or 'variational')
- `--latent_dim`: Dimension of the latent space (default: 32)
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
# Generate digits using standard autoencoder
python src/generate.py --model_type standard --num_samples 10

# Generate digits using variational autoencoder
python src/generate.py --model_type variational --num_samples 10
```

Parameters:
- `--model_type`: Type of autoencoder ('standard' or 'variational')
- `--latent_dim`: Dimension of the latent space (must match training)
- `--num_samples`: Number of digits to generate (default: 10)

## Project Structure

```
.
├── README.md
├── requirements.txt
├── checkpoints/
│   ├── best_model/    # Stores the best model for each type
│   └── run_TIMESTAMP/ # Created for each training run
└── src/
    ├── model.py       # Autoencoder model definitions
    ├── train.py       # Training script
    └── generate.py    # Generation script
```

## Output Files

Each training run creates a timestamped directory containing:
- Checkpoint files saved every 10 epochs
- Training loss plot
- Sample reconstructions during training
- Final model state

The best model for each type is maintained in:
- `checkpoints/best_model/autoencoder_standard_best.pth`
- `checkpoints/best_model/autoencoder_variational_best.pth`

Generated digits are saved with timestamps in the root directory.

## Device Support

The code automatically selects the best available device in the following order:
1. NVIDIA CUDA GPU (if available)
2. Apple Metal (MPS) for Apple Silicon/Intel Macs
3. CPU (fallback)

No additional configuration is needed - the code will automatically detect and use the best available device.