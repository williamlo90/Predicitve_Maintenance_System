import os
import sys
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    # Path untuk menyimpan preprocessing pipeline sebagai pickle
    data_processor_obj_file_path = os.path.join("artifacts", "processor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_preprocessing_pipeline(self):
        """
        Membuat preprocessing pipeline untuk data numerik dan kategorikal.
        """
        try:
            num_features = [
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]"
            ]
            cat_features = ["Type"]

            # Pipeline untuk fitur numerik
            num_pipeline = Pipeline([("scaler", StandardScaler())])

            # Pipeline untuk fitur kategorikal
            cat_pipeline = Pipeline([
                ("onehot", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Numerical features: {num_features}")
            logging.info(f"Categorical features: {cat_features}")

            preprocessor = ColumnTransformer([
                ("num", num_pipeline, num_features),
                ("cat", cat_pipeline, cat_features)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_transformation(self, train_path, test_path):
        """
        Load train/test dataset, transform using preprocessing pipeline,
        and return transformed arrays + path ke pipeline object.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("✅ Train & test data loaded successfully.")

            preprocess_obj = self.get_preprocessing_pipeline()

            # Tentukan kolom target dan yang tidak diperlukan
            target_col = "Target"
            drop_cols = ["UDI", "Product ID", "Failure Type"]

            # Pisahkan input dan target
            X_train = train_df.drop(columns=[target_col] + drop_cols)
            y_train = train_df[[target_col]]
            X_test = test_df.drop(columns=[target_col] + drop_cols)
            y_test = test_df[[target_col]]

            logging.info("🚧 Applying preprocessing pipeline to datasets...")

            X_train_transformed = preprocess_obj.fit_transform(X_train)
            X_test_transformed = preprocess_obj.transform(X_test)

            train_array = np.c_[X_train_transformed, y_train.values]
            test_array = np.c_[X_test_transformed, y_test.values]

            # Simpan objek pipeline
            save_object(self.config.data_processor_obj_file_path, preprocess_obj)
            logging.info("✅ Preprocessing pipeline saved.")

            return train_array, test_array, self.config.data_processor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
