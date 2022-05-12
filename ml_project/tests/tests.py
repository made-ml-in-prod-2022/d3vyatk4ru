
# pylint: disable=E0401, C0116, C0115, C0114, R0201

import unittest
import os

from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

from src.data import read_dataset, split_train_val_data
from src.features.make_features import (
    extract_target,
    drop_target,
    build_feature_transformer,
    make_features,
)

from src.features.custom_transformer import (
    CustomTransformer
)

from src.train_pipeline import train_pipeline
from src.predict_pipeline import predict_pipeline

from src.entity import SplittingParams
from src.entity.feature_params import FeatureParams

class TestProject(unittest.TestCase):

    def test_read_dataset(self):

        data = read_dataset('data/raw/heart_cleveland_upload.csv')

        self.assertEqual(297, len(data))

    def test_split_data(self):

        data = read_dataset('data/raw/heart_cleveland_upload.csv')

        splitting_params = SplittingParams(random_state=42, test_size=0.1)

        train, valid = split_train_val_data(data, splitting_params)

        self.assertEqual(267, len(train))
        self.assertEqual(30, len(valid))

    def test_make_features(self):

        categorical_feature = [
            'sex',
            'cp',
            'fbs',
            'restecg',
            'exang',
            'slope',
            'thal',
            'ca',
        ]

        numerical_features = [
            'age',
            'trestbps',
            'chol',
            'thalach',
            'oldpeak',
        ]

        target_col = 'condition'

        splitting_params = SplittingParams(random_state=42, test_size=0.1)

        feature_params = FeatureParams(
            categorical_features=categorical_feature,
            numerical_features=numerical_features,
            target_col=target_col
        )

        data = read_dataset('data/raw/heart_cleveland_upload.csv')

        train_df, _ = split_train_val_data(data, splitting_params)

        train_target = extract_target(
            train_df, feature_params
        )

        self.assertEqual(len(train_target), 267)

        train_df = train_df.drop(target_col, axis=1)

        self.assertTrue(target_col not in train_df.columns)

        transformer = build_feature_transformer(feature_params)

        print('############', type(transformer))

        transformer.fit(train_df)

        _ = make_features(
            transformer,
            train_df
        )

    def test_custom_transformer(self):

        categorical_feature = [
            'sex',
            'cp',
            'fbs',
            'restecg',
            'exang',
            'slope',
            'thal',
            'ca',
        ]

        numerical_features = [
            'age',
            'trestbps',
            'chol',
            'thalach',
            'oldpeak',
        ]

        target_col = 'condition'

        splitting_params = SplittingParams(random_state=42, test_size=0.1)

        feature_params = FeatureParams(
            categorical_features=categorical_feature,
            numerical_features=numerical_features,
            target_col=target_col
        )

        data = read_dataset('data/raw/heart_cleveland_upload.csv')

        train_df, _ = split_train_val_data(data, splitting_params)

        transformer = CustomTransformer(feature_params)
        transformer.fit(
            train_df
        )

        train_features = make_features(
            transformer,
            train_df[feature_params.numerical_features]
        )

        self.assertTrue(np.isclose(train_features.max(),  1))
        self.assertTrue(np.isclose(train_features.min(), 0))


    def test_train_pipeline_log_reg(self):
        train_pipeline(os.path.abspath('configs/train_config_log_reg.yaml'))

    def test_test_pipeline_log_reg(self):
        train_pipeline(os.path.abspath('configs/train_config_log_reg.yaml'))
        predict_pipeline(os.path.abspath('configs/predict_config.yaml'))


if __name__ == '__main__':

    unittest.main()
