"""
Configuration module for Fake News Detection project.
"""
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Central configuration for the project."""

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
    MODEL_DIR = BASE_DIR / "models" / "saved"
    LOGS_DIR = BASE_DIR / "logs"

    # Create directories if they don't exist
    for directory in [DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # Model Configuration
    MODEL_NAME = "distilbert-base-uncased"
    NUM_LABELS = 2  # Binary classification: Fake (0) or Real (1)
    MAX_LENGTH = 512  # Maximum sequence length

    # Training Configuration
    TRAIN_BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    TRAIN_TEST_SPLIT = 0.2
    RANDOM_SEED = 42

    # Preprocessing Configuration
    LOWERCASE = True
    REMOVE_STOPWORDS = True
    REMOVE_PUNCTUATION = True
    MIN_TEXT_LENGTH = 10  # Minimum words in article

    # API Configuration
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_TITLE = "Fake News Detection API"
    API_VERSION = "1.0.0"

    # Dataset URLs (Kaggle Fake News dataset)
    DATASET_URL = "https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset"

    # Labels
    LABEL_MAP = {
        0: "FAKE",
        1: "REAL"
    }

    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return {
            "model_name": cls.MODEL_NAME,
            "num_labels": cls.NUM_LABELS,
            "max_length": cls.MAX_LENGTH,
            "train_batch_size": cls.TRAIN_BATCH_SIZE,
            "eval_batch_size": cls.EVAL_BATCH_SIZE,
            "learning_rate": cls.LEARNING_RATE,
            "num_epochs": cls.NUM_EPOCHS,
            "random_seed": cls.RANDOM_SEED,
        }

    @classmethod
    def update_config(cls, **kwargs):
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(cls, key.upper()):
                setattr(cls, key.upper(), value)