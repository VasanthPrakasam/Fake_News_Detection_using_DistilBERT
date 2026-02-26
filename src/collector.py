"""
Data Collection Module
Handles downloading and loading of fake news datasets.
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Optional
import requests
from io import StringIO

from config.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    """Handles data collection and loading operations."""

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize DataCollector.

        Args:
            data_dir: Directory to store data files
        """
        self.data_dir = data_dir or Config.DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_kaggle_dataset(self) -> bool:
        """
        Download dataset from Kaggle.
        Note: Requires Kaggle API credentials setup.

        Returns:
            Success status
        """
        try:
            import kaggle
            logger.info("Downloading Fake News dataset from Kaggle...")

            # Download dataset
            kaggle.api.dataset_download_files(
                'clmentbisaillon/fake-and-real-news-dataset',
                path=str(self.data_dir),
                unzip=True
            )
            logger.info(f"Dataset downloaded to {self.data_dir}")
            return True

        except ImportError:
            logger.error("Kaggle API not installed. Install with: pip install kaggle")
            return False
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            return False

    def load_local_dataset(
            self,
            fake_news_path: Optional[str] = None,
            real_news_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load dataset from local files.

        Args:
            fake_news_path: Path to fake news CSV
            real_news_path: Path to real news CSV

        Returns:
            Tuple of (fake_df, real_df)
        """
        fake_path = fake_news_path or str(self.data_dir / "Fake.csv")
        real_path = real_news_path or str(self.data_dir / "True.csv")

        try:
            logger.info(f"Loading fake news from: {fake_path}")
            fake_df = pd.read_csv(fake_path)

            logger.info(f"Loading real news from: {real_path}")
            real_df = pd.read_csv(real_path)

            logger.info(f"Loaded {len(fake_df)} fake news articles")
            logger.info(f"Loaded {len(real_df)} real news articles")

            return fake_df, real_df

        except FileNotFoundError as e:
            logger.error(f"Dataset files not found: {e}")
            logger.info("Please download the dataset first or provide correct paths.")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def create_labeled_dataset(
            self,
            fake_df: pd.DataFrame,
            real_df: pd.DataFrame,
            text_column: str = 'text',
            title_column: str = 'title'
    ) -> pd.DataFrame:
        """
        Combine fake and real news into a single labeled dataset.

        Args:
            fake_df: DataFrame with fake news
            real_df: DataFrame with real news
            text_column: Name of text column
            title_column: Name of title column

        Returns:
            Combined DataFrame with labels
        """
        # Add labels
        fake_df['label'] = 0  # Fake
        real_df['label'] = 1  # Real

        # Combine title and text if both exist
        for df in [fake_df, real_df]:
            if title_column in df.columns and text_column in df.columns:
                df['combined_text'] = df[title_column] + '. ' + df[text_column]
            elif text_column in df.columns:
                df['combined_text'] = df[text_column]
            elif title_column in df.columns:
                df['combined_text'] = df[title_column]

        # Combine datasets
        combined_df = pd.concat([fake_df, real_df], ignore_index=True)

        # Shuffle
        combined_df = combined_df.sample(frac=1, random_state=Config.RANDOM_SEED).reset_index(drop=True)

        logger.info(f"Created combined dataset with {len(combined_df)} articles")
        logger.info(f"Fake news: {(combined_df['label'] == 0).sum()}")
        logger.info(f"Real news: {(combined_df['label'] == 1).sum()}")

        return combined_df[['combined_text', 'label']]

    def load_and_prepare_dataset(self) -> pd.DataFrame:
        """
        Complete pipeline: load data and prepare labeled dataset.

        Returns:
            Prepared DataFrame
        """
        fake_df, real_df = self.load_local_dataset()
        dataset = self.create_labeled_dataset(fake_df, real_df)

        # Save processed dataset
        output_path = Config.PROCESSED_DATA_DIR / "labeled_dataset.csv"
        dataset.to_csv(output_path, index=False)
        logger.info(f"Saved processed dataset to {output_path}")

        return dataset

    def create_sample_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Create a small sample dataset for quick testing.

        Args:
            n_samples: Number of samples to create

        Returns:
            Sample DataFrame
        """
        fake_samples = [
            "Breaking: Scientists discover shocking truth about vaccines causing autism",
            "You won't believe what this celebrity said about the government",
            "Miracle cure for cancer hidden by pharmaceutical companies",
            "Aliens confirmed by government officials in secret meeting",
            "This one weird trick will make you rich overnight"
        ]

        real_samples = [
            "The Federal Reserve announced a new interest rate policy today",
            "New study published in Nature shows climate change effects",
            "Technology company releases quarterly earnings report",
            "International summit discusses global trade agreements",
            "Research team develops new treatment for heart disease"
        ]

        # Replicate samples to reach n_samples
        n_per_class = n_samples // 2
        fake_texts = (fake_samples * (n_per_class // len(fake_samples) + 1))[:n_per_class]
        real_texts = (real_samples * (n_per_class // len(real_samples) + 1))[:n_per_class]

        df = pd.DataFrame({
            'combined_text': fake_texts + real_texts,
            'label': [0] * len(fake_texts) + [1] * len(real_texts)
        })

        return df.sample(frac=1, random_state=Config.RANDOM_SEED).reset_index(drop=True)


if __name__ == "__main__":
    # Example usage
    collector = DataCollector()

    # Option 1: Download from Kaggle (requires setup)
    # collector.download_kaggle_dataset()

    # Option 2: Load from local files
    # dataset = collector.load_and_prepare_dataset()

    # Option 3: Create sample dataset for testing
    sample_data = collector.create_sample_dataset(n_samples=100)
    print(sample_data.head())
    print(f"\nDataset shape: {sample_data.shape}")
    print(f"Label distribution:\n{sample_data['label'].value_counts()}")