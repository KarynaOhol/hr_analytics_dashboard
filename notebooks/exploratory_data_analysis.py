import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Load the dataset
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, 'data', 'WA_Fn-UseC_-HR-Employee-Attrition.csv')

hr_data = pd.read_csv(data_path)

# Create a directory for saving visualizations if it doesn't exist
viz_dir = os.path.join(project_root, 'visualizations')
if not os.path.exists(viz_dir):
    os.makedirs(viz_dir)


# 1. Analyze categorical variables
def analyze_categorical(df, column):
    # Count plot
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=column, data=df)
    plt.title(f'Distribution of {column}')
    plt.xticks(rotation=45)

    # Add percentage labels
    total = len(df)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / total)
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'{viz_dir}/{column}_distribution.png')
    plt.close()

    # Analyze attrition rate for this category
    attrition_rate = df.groupby(column)['Attrition'].apply(
        lambda x: (x == 'Yes').mean() * 100
    ).reset_index()
    attrition_rate.columns = [column, 'AttritionRate']

    plt.figure(figsize=(10, 6))
    sns.barplot(x=column, y='AttritionRate', data=attrition_rate)
    plt.title(f'Attrition Rate by {column}')
    plt.xticks(rotation=45)
    plt.ylabel('Attrition Rate (%)')
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/{column}_attrition.png')
    plt.close()

    return attrition_rate


# Analyze key categorical variables
categorical_columns = ['Department', 'JobRole', 'MaritalStatus', 'Gender',
                       'EducationField', 'BusinessTravel', 'OverTime']

for column in categorical_columns:
    print(f"\nAnalyzing {column}...")
    result = analyze_categorical(hr_data, column)
    print(result)


# 2. Analyze numerical variables
def analyze_numerical(df, column):
    # Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=column, kde=True)
    plt.title(f'Distribution of {column}')
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/{column}_distribution.png')
    plt.close()

    # Box plot by attrition
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Attrition', y=column, data=df)
    plt.title(f'{column} by Attrition')
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/{column}_by_attrition.png')
    plt.close()

    # Calculate statistics
    stats = df.groupby('Attrition')[column].agg(['mean', 'median', 'std']).reset_index()
    return stats


# Analyze key numerical variables
numerical_columns = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'DistanceFromHome',
                     'JobSatisfaction', 'WorkLifeBalance', 'YearsWithCurrManager']

for column in numerical_columns:
    print(f"\nAnalyzing {column}...")
    result = analyze_numerical(hr_data, column)
    print(result)

# 3. Correlation Analysis
# Select numerical columns for correlation analysis
corr_columns = ['Age', 'DailyRate', 'DistanceFromHome', 'Education',
                'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
                'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
                'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
                'RelationshipSatisfaction', 'TotalWorkingYears', 'TrainingTimesLastYear',
                'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
                'YearsSinceLastPromotion', 'YearsWithCurrManager']

# Create binary attrition column (1 for Yes, 0 for No)
hr_data['AttritionBinary'] = hr_data['Attrition'].map({'Yes': 1, 'No': 0})
corr_columns.append('AttritionBinary')

# Calculate correlation matrix
correlation_matrix = hr_data[corr_columns].corr()

# Plot correlation heatmap
plt.figure(figsize=(16, 12))
mask = np.triu(correlation_matrix)
sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm',
            center=0, linewidths=.5, cbar_kws={"shrink": .8})
plt.title('Correlation Heatmap', fontsize=18)
plt.tight_layout()
plt.savefig(f'{viz_dir}/correlation_heatmap.png')
plt.close()

# 4. Feature importance for attrition (basic)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Identify categorical and numerical columns
cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender',
            'JobRole', 'MaritalStatus', 'OverTime']
num_cols = [col for col in hr_data.columns if col not in cat_cols +
            ['Attrition', 'EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours', 'AttritionBinary']]

# Prepare the feature matrix and target vector
X = hr_data[cat_cols + num_cols]
y = hr_data['AttritionBinary']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first'), cat_cols)
    ])

# Create and train the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

model.fit(X, y)

# Get feature names after one-hot encoding
cat_features = list(model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(cat_cols))
all_features = num_cols + cat_features

# Get feature importances
importances = model.named_steps['classifier'].feature_importances_

# Create a DataFrame with feature importances
feature_importance = pd.DataFrame({
    'Feature': all_features,
    'Importance': importances
})
feature_importance = feature_importance.sort_values('Importance', ascending=False).reset_index(drop=True)

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Top 15 Feature Importances for Predicting Attrition')
plt.tight_layout()
plt.savefig(f'{viz_dir}/feature_importance.png')
plt.close()

print("\nTop 15 most important features for predicting attrition:")
print(feature_importance.head(15))

# 5. Save summary statistics to a file
summary_stats = {
    'Total Employees': len(hr_data),
    'Attrition Rate': f"{hr_data['AttritionBinary'].mean() * 100:.2f}%",
    'Average Age': f"{hr_data['Age'].mean():.2f} years",
    'Gender Ratio': f"Male: {hr_data['Gender'].value_counts()['Male'] / len(hr_data) * 100:.2f}%, "
                    f"Female: {hr_data['Gender'].value_counts()['Female'] / len(hr_data) * 100:.2f}%",
    'Average Monthly Income': f"${hr_data['MonthlyIncome'].mean():.2f}",
    'Average Years at Company': f"{hr_data['YearsAtCompany'].mean():.2f}",
    'Most Common Job Role': hr_data['JobRole'].value_counts().index[0]
}

with open(f'{viz_dir}/summary_statistics.txt', 'w') as f:
    for key, value in summary_stats.items():
        f.write(f"{key}: {value}\n")

print("\nAnalysis completed! Visualizations saved to:", viz_dir)
print("\nSummary Statistics:")
for key, value in summary_stats.items():
    print(f"{key}: {value}")
