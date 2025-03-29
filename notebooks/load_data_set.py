import pandas as pd
import os

# data_path = '/home/karina/PycharmProjects/hr_analytics_dashboard/data/WA_Fn-UseC_-HR-Employee-Attrition.csv'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the path to the dataset
data_path = os.path.join(project_root, 'data', 'WA_Fn-UseC_-HR-Employee-Attrition.csv')



hr_data = pd.read_csv(data_path)

# Display basic information about the dataset
print(f"Dataset shape: {hr_data.shape}")
print("\nFirst 5 rows:")
print(hr_data.head())

print("\nColumn names:")
print(hr_data.columns.tolist())

print("\nSummary statistics:")
print(hr_data.describe())

# Check for missing values
print("\nMissing values count:")
print(hr_data.isnull().sum())

# Check the unique values for categorical columns
print("\nUnique values in 'Attrition' column:")
print(hr_data['Attrition'].value_counts())