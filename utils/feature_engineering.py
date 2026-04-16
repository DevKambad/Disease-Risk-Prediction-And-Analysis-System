from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class FeatureEngineer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()

        if 'age' in X.columns:
            X['age_group'] = pd.cut(
                X['age'],
                bins=[0, 30, 60, 120],
                labels=['young', 'middle', 'old']
            )

        if 'bmi' in X.columns:
            X['bmi_category'] = pd.cut(
                X['bmi'],
                bins=[0, 18.5, 25, 30, 100],
                labels=['underweight', 'normal', 'overweight', 'obese']
            )

        if 'HbA1c_level' in X.columns:
            X['hba1c_risk'] = pd.cut(
                X['HbA1c_level'],
                bins=[0, 5.7, 6.5, 20],
                labels=['normal', 'prediabetes', 'diabetes']
            )

        if 'blood_glucose_level' in X.columns:
            X['glucose_risk'] = pd.cut(
                X['blood_glucose_level'],
                bins=[0, 140, 200, 500],
                labels=['normal', 'prediabetes', 'diabetes']
            )

        
        if 'chol' in X.columns:
            X['chol_level'] = pd.cut(
                X['chol'],
                bins=[0, 200, 240, 600],
                labels=['normal', 'borderline', 'high']
            )

        if 'trestbps' in X.columns:
            X['bp_category'] = pd.cut(
                X['trestbps'],
                bins=[0, 120, 140, 200],
                labels=['normal', 'elevated', 'high']
            )

        if 'thalach' in X.columns:
            X['hr_risk'] = pd.cut(
                X['thalach'],
                bins=[0, 100, 150, 220],
                labels=['low', 'medium', 'high']
            )

        if 'chol' in X.columns and 'age' in X.columns:
            X['chol_age_ratio'] = X['chol'] / (X['age'] + 1)

        if 'trestbps' in X.columns and 'age' in X.columns:
            X['bp_age_ratio'] = X['trestbps'] / (X['age'] + 1)

        if 'oldpeak' in X.columns:
            X['oldpeak_squared'] = X['oldpeak'] ** 2

        return X