""" Make custom transformer """

import logging
from typing import NoReturn

import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
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
        self.transform_numerical = StandardScaler()
        self.numerical_features = features.numerical_features

    def fit(
        self,
        data: pd.DataFrame,
        ) -> pd.DataFrame:
        """ Fitting transformer for numerical features"""
        self.transform_numerical.fit(data[self.numerical_features])
        return self

    def transform(self,
        data: pd.DataFrame,
        ):
        """ Transform numerical feature """
        return self.transform_numerical.transform(data[self.numerical_features])
