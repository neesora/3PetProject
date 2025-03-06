import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

df = pd.read_csv('data.csv')


# Calculate AverageMonthlySpend and round to 2 decimal places
df['AverageMonthlySpend'] = df['TotalCharges'] / df['Tenure']
df['AverageMonthlySpend'] = df['AverageMonthlySpend'].round(2)

# Save the updated dataframe to a CSV file
df.to_csv('data.csv', index=False)