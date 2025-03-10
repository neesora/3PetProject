import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
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
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_columns),
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), X.select_dtypes(exclude=['object']).columns)
    ],
    remainder='passthrough'
)

# Create a pipeline with the preprocessor, SMOTE, and the gradient boosting classifier
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=44)),
    ('classifier', GradientBoostingClassifier(random_state=44))
])

# Split data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44, stratify=y)

# Define hyperparameter grid for GradientBoostingClassifier
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Make predictions
predictions = best_model.predict(X_test)

# Evaluate model
print("Gradient Boosting Metrics:")
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions, zero_division=1))
print("Recall:", recall_score(y_test, predictions, zero_division=1))
print("F1-score:", f1_score(y_test, predictions, zero_division=1))
print("AUC-ROC:", roc_auc_score(y_test, predictions))