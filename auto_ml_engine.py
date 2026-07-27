import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score, precision_score, recall_score
import warnings

warnings.filterwarnings("ignore")

def detect_problem_type(y_series):
    if pd.api.types.is_numeric_dtype(y_series):
        n_unique = y_series.nunique()
        if n_unique <= 20:
            return 'classification'
        return 'regression'
    else:
        return 'classification'

def get_models(problem_type):
    if problem_type == 'regression':
        return {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'RandomForestRegressor': RandomForestRegressor(random_state=42),
            'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
            'SVR': SVR(),
            'KNeighborsRegressor': KNeighborsRegressor()
        }
    else:
        return {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=500),
            'RandomForestClassifier': RandomForestClassifier(random_state=42),
            'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
            'SVC': SVC(probability=True, random_state=42),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42)
        }

def train_all_models(X_train, y_train, X_test, y_test, problem_type):
    models = get_models(problem_type)
    results = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics = {}
            if problem_type == 'regression':
                metrics['R2'] = r2_score(y_test, y_pred)
                metrics['MAE'] = mean_absolute_error(y_test, y_pred)
                metrics['RMSE'] = np.sqrt(mean_squared_error(y_test, y_pred))
            else:
                metrics['Accuracy'] = accuracy_score(y_test, y_pred)
                metrics['F1'] = f1_score(y_test, y_pred, average='weighted')
                metrics['Precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                metrics['Recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # CV score
            if problem_type == 'regression':
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
            metrics['CV_Mean'] = cv_scores.mean()
            metrics['CV_Std'] = cv_scores.std()
            
            results[name] = {
                'model': model,
                'metrics': metrics,
                'y_pred': y_pred
            }
        except Exception as e:
            print(f"Failed to train {name}: {str(e)}")
            
    return results

def select_best_model(results, problem_type):
    best_model_name = None
    best_score = -np.inf
    
    for name, data in results.items():
        if problem_type == 'regression':
            score = data['metrics']['R2']
        else:
            score = data['metrics']['Accuracy']
            
        if score > best_score:
            best_score = score
            best_model_name = name
            
    return best_model_name, results[best_model_name]['model']

def get_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
        if len(importances.shape) > 1:
            importances = importances.mean(axis=0)
    else:
        # Model doesn't support feature importances
        return None
        
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    df = df.sort_values(by='Importance', ascending=False).head(15)
    return df
