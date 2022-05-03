""" Predicting pipline for ml model """

# pylint: disable=E0401, E0611, E1120

import logging
import sys

import click
import pandas as pd
from pandas import DataFrame

from entity.predict_pipeline_params import (
    PredictPipelineParams,
    read_predict_pipeline_params,
)

from data.make_dataset import (
    read_dataset,
)

from features.make_features import (
    drop_target,
    make_features,
)

from models.model_fit_predict import (
    predict_model,
    load_model,
    load_transformer,
)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict_pipeline(config_path: str):
    """ Predict pipeline """
    predict_pipeline_params = read_predict_pipeline_params(config_path)
    return predict_pipeline_run(predict_pipeline_params)


def predict_pipeline_run(predict_pipeline_params: PredictPipelineParams):
    """ Start pipeline """

    logger.info('Start predict pipeline with...')

    print(predict_pipeline_params)

    data = read_dataset(predict_pipeline_params.input_data_path)

    if predict_pipeline_params.target_in_dataset:
        data = drop_target(
            data,
            predict_pipeline_params
        )

    transformer = load_transformer(predict_pipeline_params.transformer_path)

    feature: DataFrame = make_features(
        transformer,
        data
    )

    model = load_model(predict_pipeline_params.model_path)

    logger.info('Making predict...')
    predict = predict_model(
        model,
        feature
    )

    logger.info('Writing to file...')
    pd.Series(predict,
              index=data.index,
              name='Predict'
              ).to_csv(predict_pipeline_params.predict_path)


@click.command(name='predict_pipeline')
@click.argument('config_path')
def predict_pipeline_command(config_path: str):
    """ Make start for terminal """
    predict_pipeline(config_path)


if __name__ == '__main__':
    predict_pipeline_command()
