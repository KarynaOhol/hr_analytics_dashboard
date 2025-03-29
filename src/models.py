import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline


def build_attrition_model(X, y, preprocessor, model_type='random_forest'):
    """
    Build and train a model to predict employee attrition
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select the model
    if model_type == 'random_forest':
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'logistic_regression':
        classifier = LogisticRegression(random_state=42, max_iter=1000)
    else:
        raise ValueError("Unsupported model type. Use 'random_forest' or 'logistic_regression'")

    # Create and train the pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }

    return model, metrics


def get_feature_names(preprocessor, cat_cols, num_cols):
    """
    Get feature names after preprocessing
    """
    cat_features = list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
    feature_names = num_cols + cat_features
    return feature_names


def predict_attrition_probability(model, X):
    """
    Predict the probability of attrition for employees
    """
    # Get probabilities
    probs = model.predict_proba(X)[:, 1]

    # Create a DataFrame with employee info and attrition probability
    results = X.copy()
    results['AttritionProbability'] = probs

    return results


def identify_high_risk_employees(df, model, preprocessor, cat_cols, num_cols, threshold=0.5):
    """
    Identify employees with high risk of attrition
    """
    # Prepare the data
    X = df[cat_cols + num_cols]

    # Get predictions
    probs = model.predict_proba(X)[:, 1]

    # Add predictions to the original data
    at_risk_df = df.copy()
    at_risk_df['AttritionProbability'] = probs
    at_risk_df['HighRisk'] = probs >= threshold

    # Sort by risk (descending)
    at_risk_df = at_risk_df.sort_values('AttritionProbability', ascending=False)

    return at_risk_df