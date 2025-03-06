#Exploratory Data Analysis
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
df = pd.read_csv('data.csv')

# Churn rate by demographics
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Churn')
plt.title('Churn Distribution')
plt.show()

categorical_vars = ['Gender', 'Geography']
numerical_vars = ['Tenure', 'MonthlyCharges', 'Age']

# Churn rate by demographics
for var in categorical_vars:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=var, hue='Churn')
    plt.title(f'Churn by {var}')
    plt.show()

# Churn rate by numerical variables
for var in numerical_vars:
    plt.figure(figsize=(10, 6))
    sns.pointplot(data=df, x='Churn', y=var)
    plt.title(f'Churn by {var}')
    plt.show()

