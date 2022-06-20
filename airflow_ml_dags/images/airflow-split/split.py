from typing import NoReturn
import click
import os

from sklearn.model_selection import train_test_split
import pandas as pd


@click.command('split')
@click.option('--splitted_path')
@click.option('--preprocessed_path')
@click.option('--random_state', default=42)
@click.option('--test_size', default=0.15)
def split(splitted_path: str,
          preprocessed_path: str,
          random_state: int,
          test_size: float) -> NoReturn:
    """ Splitting the dataset to train and test """

    data = pd.read_csv(
        os.path.join(preprocessed_path, 'syntetic_dataset.csv'),
    )

    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
    )

    os.makedirs(splitted_path, exist_ok=True)

    train.to_csv(
        os.path.join(splitted_path, 'train.csv'),
        index=False
    )

    test.to_csv(
        os.path.join(splitted_path, 'test.csv'),
        index=False
    )

if __name__ == "__main__":
    split()