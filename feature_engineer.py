import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class TimeSeriesFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, target_col=None, date_col=None, lags=[1, 2, 3, 5, 7], rolling_windows=[3, 7]):
        self.target_col = target_col
        self.date_col = date_col
        self.lags = lags
        self.rolling_windows = rolling_windows

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_out = X.copy()
        if self.target_col and self.target_col in X_out.columns:
            for lag in self.lags:
                X_out[f'{self.target_col}_lag_{lag}'] = X_out[self.target_col].shift(lag)
            for win in self.rolling_windows:
                X_out[f'{self.target_col}_rolling_mean_{win}'] = X_out[self.target_col].rolling(window=win).mean()
                X_out[f'{self.target_col}_rolling_std_{win}'] = X_out[self.target_col].rolling(window=win).std()
            
            # Fill NA created by shift/rolling
            X_out.fillna(method='bfill', inplace=True)
            X_out.fillna(method='ffill', inplace=True)
        return X_out

class AutoFeatureEngineer:
    def __init__(self, time_series=False, target_col=None, date_col=None):
        self.time_series = time_series
        self.target_col = target_col
        self.date_col = date_col
        self.pipeline = None
        self.numeric_features = []
        self.categorical_features_ohe = []
        self.categorical_features_le = []
        self.excluded_features = []
        self.label_encoders = {}
        self.feature_names_out_ = None
        
    def detect_columns(self, X):
        self.numeric_features = []
        self.categorical_features_ohe = []
        self.categorical_features_le = []
        self.excluded_features = []
        
        n_rows = len(X)
        for col in X.columns:
            if col == self.target_col:
                continue
                
            # Exclude datetime
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                self.excluded_features.append(col)
                continue
                
            # Detect high cardinality ID-like columns
            n_unique = X[col].nunique()
            if n_unique / n_rows > 0.95 and pd.api.types.is_object_dtype(X[col]):
                self.excluded_features.append(col)
                continue
                
            if pd.api.types.is_numeric_dtype(X[col]):
                self.numeric_features.append(col)
            else:
                if n_unique <= 10:
                    self.categorical_features_ohe.append(col)
                else:
                    self.categorical_features_le.append(col)
                    
    def fit_transform(self, X, y=None):
        X_copy = X.copy()
        
        if self.time_series and self.date_col:
            X_copy = X_copy.sort_values(by=self.date_col)
            
        # TS feature engineering applied outside sklearn pipeline because it requires target shifting
        if self.time_series and self.target_col in X_copy.columns:
            ts_eng = TimeSeriesFeatureEngineer(target_col=self.target_col, lags=[1, 2, 3, 5, 7], rolling_windows=[3, 7])
            X_copy = ts_eng.transform(X_copy)
            
        self.detect_columns(X_copy)
        
        # Label Encoding is handled separately because ColumnTransformer struggles with it natively without wrapping
        for col in self.categorical_features_le:
            le = LabelEncoder()
            # Convert to string and handle missing
            col_data = X_copy[col].astype(str).fillna('missing')
            X_copy[col] = le.fit_transform(col_data)
            self.label_encoders[col] = le
            
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_ohe_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        transformers = []
        if self.numeric_features:
            transformers.append(('num', numeric_transformer, self.numeric_features))
        if self.categorical_features_ohe:
            transformers.append(('cat_ohe', categorical_ohe_transformer, self.categorical_features_ohe))
            
        if self.categorical_features_le:
            # Pass through label encoded features
            transformers.append(('cat_le', 'passthrough', self.categorical_features_le))
            
        self.pipeline = ColumnTransformer(transformers=transformers, remainder='drop')
        
        transformed_X = self.pipeline.fit_transform(X_copy)
        
        # Extract feature names
        feature_names = []
        if self.numeric_features:
            feature_names.extend(self.numeric_features)
        if self.categorical_features_ohe:
            ohe = self.pipeline.named_transformers_['cat_ohe'].named_steps['onehot']
            ohe_features = ohe.get_feature_names_out(self.categorical_features_ohe)
            feature_names.extend(ohe_features)
        if self.categorical_features_le:
            feature_names.extend(self.categorical_features_le)
            
        self.feature_names_out_ = feature_names
        
        return pd.DataFrame(transformed_X, columns=self.feature_names_out_, index=X_copy.index)
        
    def transform(self, X):
        X_copy = X.copy()
        
        if self.time_series and self.date_col and self.date_col in X_copy.columns:
            X_copy = X_copy.sort_values(by=self.date_col)
            
        if self.time_series and self.target_col in X_copy.columns:
            ts_eng = TimeSeriesFeatureEngineer(target_col=self.target_col, lags=[1, 2, 3, 5, 7], rolling_windows=[3, 7])
            X_copy = ts_eng.transform(X_copy)
            
        for col in self.categorical_features_le:
            if col in X_copy.columns:
                le = self.label_encoders[col]
                col_data = X_copy[col].astype(str).fillna('missing')
                # Handle unseen labels
                classes = list(le.classes_)
                col_data = col_data.apply(lambda x: x if x in classes else classes[0])
                X_copy[col] = le.transform(col_data)
                
        transformed_X = self.pipeline.transform(X_copy)
        return pd.DataFrame(transformed_X, columns=self.feature_names_out_, index=X_copy.index)

    def get_feature_names_out(self):
        return self.feature_names_out_
