import os
import click
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@click.command('predict')
@click.option("--input_path")
@click.option("--pred_path")
@click.option("--scaler_path")
@click.option("--model_path")
def predict(input_path: str, pred_path: str, scaler_path: str, model_path: str) -> None:
    """ Make predict for data """

    model = load_obj_pkl(os.path.join(model_path, 'model.pkl'))

    transformer = load_obj_pkl(os.path.join(scaler_path, 'transformer.pkl'))

    data = pd.read_csv(
        os.path.join(input_path, 'test.csv'),
        index_col=0
    )

    data = transformer.transform(data)
    prediction = model.predict(data)

    os.makedirs(pred_path, exist_ok=True)

    np.savetxt(os.path.join(pred_path, 'predictions.csv'), prediction, delimiter=",")

def load_obj_pkl(path: str):
    """ Load pickle object """

    with open(path, 'rb') as file:
        obj = pickle.load(file)
        return obj

if __name__ == '__main__':
    predict()