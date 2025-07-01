import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# -------------------------------
# 1. Custom Transformer: Aggregate per customer
# -------------------------------
class CustomerAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregates transaction-level data into customer-level summary statistics.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        agg_df = df.groupby('CustomerId').agg(
            total_amount=('Amount', 'sum'),
            avg_amount=('Amount', 'mean'),
            transaction_count=('Amount', 'count'),
            std_amount=('Amount', 'std')
        ).reset_index()
        return agg_df


# -------------------------------
# 2. Custom Transformer: Time-based Feature Extraction
# -------------------------------
class TimeFeaturesExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts hour, day, month, and year from 'TransactionStartTime'.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

        # Extract temporal features
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionYear'] = df['TransactionStartTime'].dt.year

        return df


# -------------------------------
# 3. Build Feature Pipeline
# -------------------------------
def build_feature_pipeline():
    """
    Constructs and returns the full sklearn pipeline for data preprocessing.
    """
    # Select features by dtype
    categorical_features = [
        'CurrencyCode', 'ProviderId', 'ProductId',
        'ProductCategory', 'ChannelId'
    ]
    
    numerical_features = [
        'Amount', 'Value', 'TransactionHour',
        'TransactionDay', 'TransactionMonth', 'TransactionYear'
    ]

    # Numeric transformer with imputation and scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Categorical transformer with imputation and one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine both transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Full pipeline: time features first, then preprocessing
    pipeline = Pipeline(steps=[
        ('time_features', TimeFeaturesExtractor()),
        ('preprocessor', preprocessor)
    ])

    return pipeline
