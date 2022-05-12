""" Make custom transformer """

import logging
from typing import NoReturn
from copy import deepcopy

import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
)

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)

# from ..entity.custom_transformer_params import TransformerParams


logger = logging.getLogger(__name__)


class CustomTransformer(BaseEstimator, TransformerMixin):
    """ Custom transformer class """
    def __init__(self, features) -> NoReturn:
        """ Class ctor """
        self.transform_numerical = MinMaxScaler()
        self.numerical_features = features.numerical_features
        self.fitted = False

    def fit(
        self,
        data: pd.DataFrame,
        ) -> pd.DataFrame:
        """ Fitting transformer for numerical features"""
        self.transform_numerical.fit(data[self.numerical_features])
        return self

    def is_fitted(self):
        """ Check fit transformer """
        if not self.fitted:
            raise Exception('CustomTransformer not fitted!')

    def transform(self,
        data: pd.DataFrame,
        ):
        """ Transform numerical feature """
        self.is_fitted()
        copy_data = deepcopy(data[self.numerical_features])
        return self.transform_numerical.transform(copy_data)
