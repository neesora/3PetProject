import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load dataset
df = pd.read_csv('data.csv')

# Split data into features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

# Train Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000, random_state=44).fit(X_train, y_train)


# Make predictions
logistic_predictions = logistic_model.predict(X_test[:2, :])

# Evaluate model
print("Logistic Regression Metrics:")
print("Accuracy:", accuracy_score(y_test, logistic_predictions))
print("Precision:", precision_score(y_test, logistic_predictions))
print("Recall:", recall_score(y_test, logistic_predictions))
print("F1-score:", f1_score(y_test, logistic_predictions))
print("AUC-ROC:", roc_auc_score(y_test, logistic_predictions))
