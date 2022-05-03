""" Make fir and predict model """
import pickle
import sys
import logging
from typing import Dict, Union, NoReturn

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

# from ..entity.train_params import TrainingParams

SklearnClassificationModel = Union[LogisticRegression, RandomForestClassifier]


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_model(features: pd.DataFrame,
                target: pd.Series,
                train_params,
                ) -> SklearnClassificationModel:
    """ Make training model """
    logger.info('Start loading %s model...', train_params.model_type)

    if train_params.model_type == 'LogisticRegression':
        model = LogisticRegression()
    elif train_params.model_type == 'RandomForestClassifier':
        model = RandomForestClassifier()
    else:
        logger.exception('Selected model is incorrect')
        raise NotImplementedError()

    logger.info('Finished loading model!')
    logger.info('Start model fitting...')

    model.fit(features, target)

    logger.info('Finished model fitting!')

    return model


def predict_model(model: SklearnClassificationModel,
                  feature: pd.DataFrame
                  ) -> np.ndarray:
    """ Make predict model """
    logger.info('Start model predict...')

    predict = model.predict(feature)

    logger.info('Finished model predict!')

    return predict


def evaluate_model(predict: np.ndarray,
                   target: pd.Series
                   ) -> Dict[str, float]:
    """ Make evaluate model """

    logger.info('Start calculate metrics for model...')

    return {
        'acc': accuracy_score(target, predict),
        'f1': f1_score(target, predict, average='macro'),
        'roc_auc': roc_auc_score(target, predict),
    }


def save_model(model: SklearnClassificationModel,
               path: str
               ) -> NoReturn:
    """ Save model to file """
    with open(path, 'wb') as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info('Model saved!')


def load_model(path: str):
    """ Load model from path """
    logger.info('Model start loading...')
    with open(path, 'rb') as model:
        return pickle.load(model)


def save_transformer(transformer, path: str):
    """ Save transformer from path """
    with open(path, 'wb') as file:
        pickle.dump(transformer, file, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info('Transformer saved!')


def load_transformer(path: str):
    """ Load transformer from path """
    logger.info('Transformer start loading...')
    with open(path, 'rb') as transformer:
        return pickle.load(transformer)
