import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

# Load dataset
df = pd.read_csv('data.csv')

# Replace inf values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Split data into features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns

# Create a column transformer with one-hot encoding for categorical columns and imputation for missing values
preprocessor = ColumnTransformer(
    transformers=[
        # Transformation for categorical columns
        ('cat', Pipeline(steps=[
            # Impute missing values in categorical columns with the most frequent value
            ('imputer', SimpleImputer(strategy='most_frequent')),
            # Encode categorical columns into binary vectors, 
            # handle_unknown='ignore' ignores unknown categories in the test set
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_columns),
        # Impute missing values in numerical columns with the mean(avg value between the min and max)
        ('num', SimpleImputer(strategy='mean'), X.select_dtypes(exclude=['object']).columns)
    ],
    # Other columns that are not transformed should be dropped
    remainder='passthrough'
)

# Create a pipeline with the preprocessor and the logistic regression model
pipeline = Pipeline(steps=[
    # Apply the preprocessor
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=44))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
logistic_predictions = pipeline.predict(X_test)

# Evaluate model
print("Logistic Regression Metrics:")
print("Accuracy:", accuracy_score(y_test, logistic_predictions))
print("Precision:", precision_score(y_test, logistic_predictions))
print("Recall:", recall_score(y_test, logistic_predictions))
print("F1-score:", f1_score(y_test, logistic_predictions))
print("AUC-ROC:", roc_auc_score(y_test, logistic_predictions))
