from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict
import sys
import logging
from typing import Optional, Dict
from colorama import Fore, Back, Style


class ColoredFormatter(logging.Formatter):
    """Colored log formatter."""

    def __init__(self, *args, colors: Optional[Dict[str, str]] = None, **kwargs) -> None:
        """Initialize the formatter with specified format strings."""

        super().__init__(*args, **kwargs)

        self.colors = colors if colors else {}

    def format(self, record) -> str:
        """Format the specified record as text."""

        record.color = self.colors.get(record.levelname, '')
        record.reset = Style.RESET_ALL

        return super().format(record)


def set_logger(level=logging.INFO):
    formatter = ColoredFormatter(
        '{color}[{levelname:.1s}] {message}{reset}',
        style='{', datefmt='%Y-%m-%d %H:%M:%S',
        colors={
            'DEBUG': Fore.CYAN,
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT,
        }
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.handlers[:] = []
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


CURRENT_DIR = Path(__file__).parent


def read_splits(*, dataset_name):
    DATA_DIR = CURRENT_DIR.parent / dataset_name
    TRAIN_FILE = DATA_DIR / "train.csv"
    DEV_FILE = DATA_DIR / "dev.csv"
    TEST_FILE = DATA_DIR / "test.csv"

    train_df, dev_df, test_df = map(
        pd.read_csv, (TRAIN_FILE, DEV_FILE, TEST_FILE)
    )
    train, dev, test = map(Dataset.from_pandas, (train_df, dev_df, test_df))
    return DatasetDict(train=train, dev=dev, test=test)