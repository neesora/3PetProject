# Import necessary libraries
import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv('data.csv')

# Check for missing values
print(df.isnull().sum())

# Fill missing values in numerical columns with median
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# Fill missing values in categorical columns with mode
catorigal_columns = df.select_dtypes(include=['object']).columns
df[catorigal_columns] = df[catorigal_columns].fillna(df[catorigal_columns].mode().iloc[0])

# Check for duplicates
print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()
