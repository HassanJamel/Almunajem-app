import pandas as pd
import numpy as np
import joblib

def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_single(model, pipeline, input_dict):
    df = pd.DataFrame([input_dict])
    X_processed = pipeline.transform(df)
    prediction = model.predict(X_processed)
    return prediction[0]

def predict_batch(model, pipeline, dataframe):
    X_processed = pipeline.transform(dataframe)
    predictions = model.predict(X_processed)
    return predictions

def get_prediction_confidence(model, X_processed):
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_processed)
        # Return max probability as confidence
        return np.max(proba, axis=1)
    return None
