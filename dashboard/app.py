import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your modules
from src.data_processing import load_data, prepare_data_for_modeling
from src.models import build_attrition_model, get_feature_names, identify_high_risk_employees
from src.metrics import compute_hr_metrics, calculate_attrition_risk_factors
from src.visualization import (create_attrition_overview, create_satisfaction_dashboard,
                               create_salary_analysis, create_attrition_prediction_chart,
                               create_tenure_analysis)

# Load the data
data_path = '/home/karina/PycharmProjects/hr_analytics_dashboard/data/WA_Fn-UseC_-HR-Employee-Attrition.csv'
df = load_data(data_path)

# Prepare data for modeling
X, y, preprocessor, cat_cols, num_cols = prepare_data_for_modeling(df)

# Build the attrition model
model, model_metrics = build_attrition_model(X, y, preprocessor)

# Get feature names
feature_names = get_feature_names(preprocessor, cat_cols, num_cols)

# Calculate feature importance
feature_importance = calculate_attrition_risk_factors(model.named_steps['classifier'], feature_names)

# Compute HR metrics
hr_metrics = compute_hr_metrics(df)

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Define app layout
app.layout = html.Div([
    html.H1("HR Analytics Dashboard", style={'textAlign': 'center', 'margin': '20px'}),

    html.Div([
        html.Div([
            html.H3("Key Metrics"),
            html.Div([
                html.Div([
                    html.H4("Total Employees"),
                    html.H2(f"{hr_metrics['employee_count']}")
                ], className="metric-card"),
                html.Div([
                    html.H4("Attrition Rate"),
                    html.H2(f"{hr_metrics['attrition_rate']:.1f}%")
                ], className="metric-card"),
                html.Div([
                    html.H4("Avg Monthly Income"),
                    html.H2(f"${hr_metrics['avg_salary']:.0f}")
                ], className="metric-card"),
                html.Div([
                    html.H4("Avg Age"),
                    html.H2(f"{hr_metrics['avg_age']:.1f}")
                ], className="metric-card")
            ], style={'display': 'flex', 'justifyContent': 'space-between'})
        ], style={'margin': '20px', 'padding': '20px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'}),
    ]),

    # Tabs for different visualizations
    dcc.Tabs([
        dcc.Tab(label="Attrition Overview", children=[
            dcc.Graph(id="attrition-overview", figure=create_attrition_overview(df))
        ]),
        dcc.Tab(label="Satisfaction Analysis", children=[
            dcc.Graph(id="satisfaction-dashboard", figure=create_satisfaction_dashboard(df))
        ]),
        dcc.Tab(label="Salary Analysis", children=[
            dcc.Graph(id="salary-analysis", figure=create_salary_analysis(df))
        ]),
        dcc.Tab(label="Tenure Analysis", children=[
            dcc.Graph(id="tenure-analysis", figure=create_tenure_analysis(df))
        ]),
        dcc.Tab(label="Attrition Prediction", children=[
            html.Div([
                html.H3("Model Performance"),
                html.Div([
                    html.Div([
                        html.H4("Accuracy"),
                        html.H2(f"{model_metrics['accuracy']:.2f}")
                    ], className="metric-card"),
                    html.Div([
                        html.H4("Precision"),
                        html.H2(f"{model_metrics['precision']:.2f}")
                    ], className="metric-card"),
                    html.Div([
                        html.H4("Recall"),
                        html.H2(f"{model_metrics['recall']:.2f}")
                    ], className="metric-card"),
                    html.Div([
                        html.H4("F1 Score"),
                        html.H2(f"{model_metrics['f1']:.2f}")
                    ], className="metric-card")
                ], style={'display': 'flex', 'justifyContent': 'space-between'})
            ], style={'margin': '20px', 'padding': '20px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'}),
            dcc.Graph(id="feature-importance", figure=create_attrition_prediction_chart(feature_importance))
        ])
    ])
])

# Add CSS styling
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        <title>HR Analytics Dashboard</title>
        {%metas%}
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                background-color: #f5f5f5;
            }
            .metric-card {
                background-color: white;
                border-radius: 8px;
                padding: 15px;
                width: 22%;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }
            .metric-card h4 {
                margin: 0;
                color: #666;
            }
            .metric-card h2 {
                margin: 10px 0 0 0;
                color: #333;
            }
            .dash-tab {
                padding: 15px 20px;
            }
            .dash-tab--selected {
                background-color: #e3f2fd;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

# Run the app
if __name__ == '__main__':
    app.run(debug=True)