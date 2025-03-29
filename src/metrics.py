import pandas as pd
import numpy as np


def compute_hr_metrics(df):
    """
    Compute key HR metrics from the dataset
    """
    metrics = {}

    # Basic metrics
    metrics['employee_count'] = len(df)
    metrics['attrition_rate'] = df['AttritionBinary'].mean() * 100
    metrics['avg_age'] = df['Age'].mean()
    metrics['avg_salary'] = df['MonthlyIncome'].mean()

    # Department metrics
    dept_counts = df['Department'].value_counts()
    metrics['department_distribution'] = {dept: count / len(df) * 100 for dept, count in dept_counts.items()}

    dept_attrition = df.groupby('Department')['AttritionBinary'].mean() * 100
    metrics['department_attrition'] = dept_attrition.to_dict()

    # Salary metrics
    metrics['salary_stats'] = {
        'min': df['MonthlyIncome'].min(),
        'max': df['MonthlyIncome'].max(),
        'median': df['MonthlyIncome'].median(),
        'q1': df['MonthlyIncome'].quantile(0.25),
        'q3': df['MonthlyIncome'].quantile(0.75)
    }

    # Satisfaction metrics (1-4 scale)
    satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction',
                         'WorkLifeBalance', 'RelationshipSatisfaction']

    metrics['satisfaction'] = {}
    for col in satisfaction_cols:
        metrics['satisfaction'][col] = df[col].mean()

    # Performance metrics
    performance_cols = ['PerformanceRating', 'JobInvolvement']
    metrics['performance'] = {}
    for col in performance_cols:
        metrics['performance'][col] = df[col].mean()

    # Job role metrics
    role_attrition = df.groupby('JobRole')['AttritionBinary'].mean() * 100
    metrics['job_role_attrition'] = role_attrition.to_dict()

    # Age group analysis
    df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 65],
                            labels=['18-25', '26-35', '36-45', '46-55', '56-65'])
    age_attrition = df.groupby('AgeGroup', observed=True)['AttritionBinary'].mean() * 100
    metrics['age_group_attrition'] = age_attrition.to_dict()

    # Overtime impact
    overtime_attrition = df.groupby('OverTime')['AttritionBinary'].mean() * 100
    metrics['overtime_attrition'] = overtime_attrition.to_dict()

    return metrics


def calculate_attrition_risk_factors(model, feature_names):
    """
    Extract attrition risk factors from a trained model
    """
    # Get feature importances from model
    importances = model.feature_importances_

    # Create a DataFrame
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)

    return feature_importance


def calculate_turnover_cost(df, cost_per_employee=10000):
    """
    Estimate turnover cost based on attrition
    Note: cost_per_employee is a placeholder - should be customized based on industry/role
    """
    attrition_count = df['AttritionBinary'].sum()
    total_cost = attrition_count * cost_per_employee

    # By department
    dept_attrition = df.groupby('Department')['AttritionBinary'].sum()
    dept_cost = dept_attrition * cost_per_employee

    return {
        'total_turnover_cost': total_cost,
        'department_turnover_cost': dept_cost.to_dict()
    }