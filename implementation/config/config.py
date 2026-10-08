from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    # Reproducibility
    seed: int = 42

    # Model
    gru_units: int = 100
    seq_len: int = 12
    pre_len: int = 1

    # Training
    learning_rate: float = 0.005
    training_epoch: int = 1
    batch_size: int = 8

    # Dataset split
    train_rate: float = 0.9

    # Regularization
    lambda_loss: float = 0.0015