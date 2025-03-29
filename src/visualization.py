import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def set_plotting_style():
    """Set consistent style for matplotlib plots"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("viridis")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12


def create_attrition_overview(df):
    """Create overview of attrition statistics"""
    # Create a subplot with 2 rows and 2 columns
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Attrition Distribution', 'Attrition by Department',
                        'Attrition by Job Role', 'Attrition by Age Group'),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "bar", "colspan": 2}, None]]
    )

    # 1. Attrition Distribution (Pie Chart)
    attrition_counts = df['Attrition'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=attrition_counts.index,
            values=attrition_counts.values,
            hole=0.3,
            marker=dict(colors=['#2ecc71', '#e74c3c']),
            textinfo='percent+label'
        ),
        row=1, col=1
    )

    # 2. Attrition by Department (Bar Chart)
    dept_attrition = df.groupby('Department')['AttritionBinary'].mean() * 100
    fig.add_trace(
        go.Bar(
            x=dept_attrition.index,
            y=dept_attrition.values,
            marker_color='#3498db',
            text=[f"{val:.1f}%" for val in dept_attrition.values],
            textposition='auto'
        ),
        row=1, col=2
    )

    # 3. Attrition by Job Role (Bar Chart)
    role_attrition = df.groupby('JobRole')['AttritionBinary'].mean() * 100
    role_attrition = role_attrition.sort_values(ascending=False)
    fig.add_trace(
        go.Bar(
            x=role_attrition.index,
            y=role_attrition.values,
            marker_color='#9b59b6',
            text=[f"{val:.1f}%" for val in role_attrition.values],
            textposition='auto'
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        height=800,
        title_text="Employee Attrition Overview",
        showlegend=False
    )
    fig.update_yaxes(title_text='Attrition Rate (%)', row=1, col=2)
    fig.update_yaxes(title_text='Attrition Rate (%)', row=2, col=1)
    fig.update_xaxes(tickangle=45, row=2, col=1)

    return fig


def create_satisfaction_dashboard(df):
    """Create a dashboard for satisfaction metrics"""
    # Create figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Job Satisfaction', 'Environment Satisfaction',
                        'Work-Life Balance', 'Relationship Satisfaction')
    )

    # Satisfaction metrics
    metrics = ['JobSatisfaction', 'EnvironmentSatisfaction',
               'WorkLifeBalance', 'RelationshipSatisfaction']

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for metric, pos in zip(metrics, positions):
        # Count by satisfaction level and attrition
        grouped = df.groupby([metric, 'Attrition']).size().reset_index()
        grouped.columns = [metric, 'Attrition', 'Count']

        # Pivot table for stacked bar chart
        pivot_data = grouped.pivot(index=metric, columns='Attrition', values='Count').fillna(0)

        # Plot stacked bar chart
        fig.add_trace(
            go.Bar(
                x=pivot_data.index,
                y=pivot_data['Yes'],
                name='Left Company',
                marker_color='#e74c3c'
            ),
            row=pos[0], col=pos[1]
        )

        fig.add_trace(
            go.Bar(
                x=pivot_data.index,
                y=pivot_data['No'],
                name='Stayed',
                marker_color='#2ecc71'
            ),
            row=pos[0], col=pos[1]
        )

    # Update layout
    fig.update_layout(
        height=800,
        title_text="Employee Satisfaction Analysis",
        barmode='stack',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update axes
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_yaxes(title_text='Number of Employees', row=i, col=j)
            fig.update_xaxes(title_text='Satisfaction Level (1-4)', row=i, col=j)

    return fig


def create_salary_analysis(df):
    """Create salary analysis visualizations"""
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Monthly Income Distribution', 'Income by Job Role',
                        'Income vs. Years at Company', 'Income by Department'),
        specs=[[{"type": "histogram"}, {"type": "box"}],
               [{"type": "scatter"}, {"type": "box"}]]
    )

    # 1. Monthly Income Distribution
    fig.add_trace(
        go.Histogram(
            x=df['MonthlyIncome'],
            nbinsx=30,
            marker_color='#3498db',
            opacity=0.7
        ),
        row=1, col=1
    )

    # 2. Income by Job Role
    fig.add_trace(
        go.Box(
            y=df['MonthlyIncome'],
            x=df['JobRole'],
            marker_color='#9b59b6'
        ),
        row=1, col=2
    )

    # 3. Income vs. Years at Company
    fig.add_trace(
        go.Scatter(
            x=df['YearsAtCompany'],
            y=df['MonthlyIncome'],
            mode='markers',
            marker=dict(
                color=df['JobLevel'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Job Level')
            )
        ),
        row=2, col=1
    )

    # 4. Income by Department
    fig.add_trace(
        go.Box(
            y=df['MonthlyIncome'],
            x=df['Department'],
            marker_color='#e67e22'
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        height=800,
        title_text="Salary Analysis Dashboard",
        showlegend=False
    )

    # Update axes
    fig.update_xaxes(title_text='Monthly Income ($)', row=1, col=1)
    fig.update_yaxes(title_text='Count', row=1, col=1)

    fig.update_xaxes(title_text='Job Role', row=1, col=2)
    fig.update_yaxes(title_text='Monthly Income ($)', row=1, col=2)

    fig.update_xaxes(title_text='Years at Company', row=2, col=1)
    fig.update_yaxes(title_text='Monthly Income ($)', row=2, col=1)

    fig.update_xaxes(title_text='Department', row=2, col=2)
    fig.update_yaxes(title_text='Monthly Income ($)', row=2, col=2)
    fig.update_xaxes(tickangle=45, row=1, col=2)

    return fig


def create_attrition_prediction_chart(feature_importance_df):
    """Create a chart of feature importance for attrition prediction"""
    # Sort the feature importance DataFrame
    sorted_df = feature_importance_df.sort_values('Importance', ascending=False).head(15)

    # Create the bar chart
    fig = go.Figure(
        go.Bar(
            x=sorted_df['Importance'],
            y=sorted_df['Feature'],
            orientation='h',
            marker_color='#3498db'
        )
    )

    # Update layout
    fig.update_layout(
        title='Top 15 Factors Influencing Employee Attrition',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=600,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def create_tenure_analysis(df):
    """Create visualizations for employee tenure analysis"""
    # Create figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Years at Company', 'Years in Current Role',
                        'Years Since Last Promotion', 'Years with Current Manager')
    )

    # Tenure metrics
    metrics = ['YearsAtCompany', 'YearsInCurrentRole',
               'YearsSinceLastPromotion', 'YearsWithCurrManager']

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for metric, pos in zip(metrics, positions):
        # Create two histograms for each metric (one for each attrition status)
        for attrition, color in zip(['Yes', 'No'], ['#e74c3c', '#2ecc71']):
            subset = df[df['Attrition'] == attrition]

            fig.add_trace(
                go.Histogram(
                    x=subset[metric],
                    name=f"{attrition} - {metric}",
                    opacity=0.7,
                    marker_color=color
                ),
                row=pos[0], col=pos[1]
            )

    # Update layout
    fig.update_layout(
        height=800,
        title_text="Employee Tenure Analysis",
        barmode='overlay'
    )

    # Update axes
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_yaxes(title_text='Number of Employees', row=i, col=j)
            fig.update_xaxes(title_text='Years', row=i, col=j)

    return fig