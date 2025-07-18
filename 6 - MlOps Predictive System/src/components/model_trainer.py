import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelConfig:
    trained_model_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("📊 Splitting train and test data")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Define candidate models
            models = {
                "Random Forest": RandomForestClassifier(),
                "SVM": SVC(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "KNN": KNeighborsClassifier()
            }

            # Hyperparameter grid
            params = {
                "Random Forest": {
                    'n_estimators': [10, 50, 100],
                    'max_depth': [None, 10, 20]
                },
                "SVM": {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [50, 100]
                },
                "Logistic Regression": {
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear', 'saga']
                },
                "KNN": {
                    'n_neighbors': [3, 5, 10],
                    'weights': ['uniform', 'distance']
                }
            }

            logging.info("🔍 Starting model evaluation with hyperparameter tuning...")

            model_scores: dict = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            best_score = max(model_scores.values())
            best_model_name = max(model_scores, key=model_scores.get)
            best_model = models[best_model_name]

            logging.info(f"🏆 Best model selected: {best_model_name} (score: {best_score})")

            # Save model
            save_object(
                file_path=self.config.trained_model_path,
                obj=best_model
            )

            # Final evaluation on test data
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            logging.info(f"✅ Final model accuracy on test set: {accuracy}")
            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
