import os
import click

import pandas as pd


@click.command('preprocessor')
@click.option('--in_path')
@click.option('--tmp_path')
def preprocessor_files(in_path: str, tmp_path: str) -> None:
    """ Preprocessor files with data """

    features = pd.read_csv(
        os.path.join(in_path, 'data.csv'),
        index_col=0,
    )
    
    target = pd.read_csv(
        os.path.join(in_path, 'target.csv'),
        index_col=0,
    )

    syntetic_dataset = features.merge(target, right_index=True, left_index=True)
    os.makedirs(tmp_path, exist_ok=True)
    syntetic_dataset.to_csv(os.path.join(tmp_path, 'syntetic_dataset.csv'))


if __name__ == "__main__":
    preprocessor_files()