# GAN Implementation

This is an implementation of a Generative Adversarial Network (GAN) trained on the MNIST dataset.

## Directory Structure

```
.
├── src/
│   ├── model.py      # Network architectures (Generator and Discriminator)
│   ├── train.py      # Training loop and utilities
│   └── generate.py   # Sample generation script
├── checkpoints/      # Saved model states
│   ├── best_model/   # Best model checkpoint
│   └── run_*/        # Timestamped training runs
├── data/            # Dataset directory
└── generated_samples/  # Output directory for generated images
```

## Architecture

The implementation consists of two networks:

- Generator: Takes random noise (latent vectors) as input and generates 28x28 grayscale images
- Discriminator: Takes 28x28 grayscale images as input and outputs a probability of the image being real

## Usage

### Training

To train the model:

```bash
python src/train.py [OPTIONS]
```

Options:
- `--latent_dim`: Dimension of the latent space (default: 100)
- `--epochs`: Number of training epochs (default: 200)
- `--batch_size`: Batch size for training (default: 128)
- `--lr`: Learning rate (default: 0.0002)
- `--beta1`: Beta1 parameter for Adam optimizer (default: 0.5)
- `--save_interval`: Save model every N epochs (default: 10)
- `--sample_interval`: Generate samples every N epochs (default: 5)
- `--early_stopping_patience`: Number of epochs to wait before early stopping (default: 10)

The training script will:
1. Create a timestamped run directory in `checkpoints/run_*/`
2. Save regular checkpoints in the run directory
3. Save the best model (based on generator loss) to `checkpoints/best_model/gan_best.pth`
4. Save generated samples to the run directory
5. Implement early stopping if the generator loss doesn't improve for the specified patience

### Generating Samples

To generate samples from the trained model:

```bash
python src/generate.py [OPTIONS]
```

Options:
- `--latent_dim`: Dimension of latent space (default: 100)
- `--n_samples`: Number of samples to generate (default: 25)
- `--interpolate`: Generate latent space interpolation (default: False)
- `--num_steps`: Number of interpolation steps (default: 10)

The script will automatically:
1. Load the best model from `checkpoints/best_model/gan_best.pth`
2. Generate either a grid of random samples or a latent space interpolation
3. Save the output with a timestamp in the `generated_samples/` directory

## Output

- Model checkpoints are saved in the `checkpoints/` directory
- The best model is saved in `checkpoints/best_model/gan_best.pth`
- Each training run creates a timestamped directory in `checkpoints/run_*/`
- Generated samples are saved in the `generated_samples/` directory with timestamps

## Requirements

- PyTorch >= 1.7.0
- torchvision >= 0.8.0
- numpy >= 1.19.0
- matplotlib >= 3.3.0
- tqdm >= 4.50.0 