import os
from typing import NoReturn
import click
import pickle

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@click.command('preprocessing')
@click.option('--scaler_path')
@click.option('--tmp_path')
@click.option('--preprocess_path')
def preprocess(scaler_path: str, tmp_path: str, preprocess_path: str) -> NoReturn:
    """ Make data preprocessing """

    data = pd.read_csv(
        os.path.join(tmp_path, 'syntetic_dataset.csv'),
        index_col=0,
    )

    target = data.target

    features = data.drop(['target'], axis=1)

    scaler = MinMaxScaler()
    
    features_df = pd.DataFrame(
        scaler.fit_transform(features)
    )

    features_df.columns = features.columns.tolist()

    data_preprocessing = features_df.merge(
        target,
        right_index=True,
        left_index=True
    )

    os.makedirs(preprocess_path, exist_ok=True)

    data_preprocessing.to_csv(
        os.path.join(preprocess_path, 'syntetic_dataset.csv'),
        index=False,
    )

    os.makedirs(scaler_path, exist_ok=True)

    save_object_pkl(scaler, scaler_path)

def save_object_pkl(obj, path: str) -> NoReturn:
    """ Save pickle object """

    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'transformer.pkl'), 'wb') as file:
        pickle.dump(obj, file)

if __name__ == '__main__':
    preprocess()