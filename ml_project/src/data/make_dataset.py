""" Subpackage for load data"""

import logging
from typing import Tuple
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

# from ..entity.split_params import SplittingParams


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def read_dataset(path: str) -> pd.DataFrame:
    """ Read dataset from csv file """

    logger.info('Loading dataset from %s...', path)

    data = pd.read_csv(path)

    logger.info('Finished loading dataset from %s!', path)
    logger.info('The dataset has %s size', data.shape)

    return data


def split_train_val_data(data: pd.DataFrame,
                         params,
                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Split data to train and validation """

    logger.info('Splitting dataset to train and test...')

    train_data, test_data = train_test_split(
        data,
        test_size=params.test_size,
        random_state=params.random_state,
    )

    logger.info('Finished splitting dataset!')

    return train_data, test_data
