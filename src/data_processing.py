import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(file_path):
    """
    Load and preprocess the HR analytics data
    """
    # Load the data
    df = pd.read_csv(file_path)

    # Create binary attrition column (1 for Yes, 0 for No)
    df['AttritionBinary'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    return df


def get_preprocessor(df):
    """
    Create a sklearn preprocessor for the HR data
    """
    # Identify categorical and numerical columns
    cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender',
                'JobRole', 'MaritalStatus', 'OverTime']
    num_cols = [col for col in df.columns if col not in cat_cols +
                ['Attrition', 'EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours', 'AttritionBinary']]

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first'), cat_cols)
        ])

    return preprocessor, cat_cols, num_cols


def prepare_data_for_modeling(df):
    """
    Prepare data for machine learning models
    """
    # Get preprocessor
    preprocessor, cat_cols, num_cols = get_preprocessor(df)

    # Prepare features and target
    X = df[cat_cols + num_cols]
    y = df['AttritionBinary']

    return X, y, preprocessor, cat_cols, num_cols


def calculate_attrition_rate(df, column=None):
    """
    Calculate attrition rate overall or by a specific column
    """
    if column is None:
        return df['AttritionBinary'].mean() * 100

    attrition_rate = df.groupby(column)['AttritionBinary'].mean() * 100
    return attrition_rate.reset_index().rename(columns={'AttritionBinary': 'AttritionRate'})


def calculate_average_tenure(df):
    """
    Calculate average tenure metrics
    """
    tenure_metrics = {
        'avg_years_at_company': df['YearsAtCompany'].mean(),
        'avg_years_in_role': df['YearsInCurrentRole'].mean(),
        'avg_years_with_manager': df['YearsWithCurrManager'].mean(),
        'avg_years_since_promotion': df['YearsSinceLastPromotion'].mean()
    }
    return tenure_metrics


def get_satisfaction_metrics(df):
    """
    Extract and summarize satisfaction metrics
    """
    satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction',
                         'WorkLifeBalance', 'RelationshipSatisfaction']

    satisfaction_summary = {}
    for col in satisfaction_cols:
        satisfaction_summary[col] = {
            'mean': df[col].mean(),
            'by_attrition': df.groupby('Attrition')[col].mean().to_dict()
        }

    return satisfaction_summary