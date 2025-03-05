import pandas as pd
import numpy as np

num_customers = 10000
#Set the seed for reproducibility
np.random.seed(44)

#Generate random data for a fictional bank with 10,000 customers
#Customer ID is a sequence from 1 to 10,000
customer_id = np.arange(1, num_customers + 1)
#Gender
gender = np.random.choice(['Male', 'Female'], size=num_customers)
#Age is a random integer between 18 and 70
age = np.random.randint(18, 71, size=num_customers)
#Geography is either Germany, France, or Spain
geography = np.random.choice(['Germany', 'France', 'Spain'], size=num_customers)
#Monthly charges is a random integer between 20 and 100
monthly_charges = np.random.randint(20, 100, size=num_customers)
#Tenure is a random integer between 0 and 10
tensure = np.random.randint(0, 11, size=num_customers)
#Balance is a random float between 0 and 200,000
balance = np.round(np.random.uniform(0, 200000, size=num_customers), 2)
#Number of products is a random integer between 1 and 3
num_of_products = np.random.randint(1, 4, size=num_customers)
#Has credit card is a binary variable with 1 being yes and 0 being no
has_card = np.random.choice([0, 1], size=num_customers)
#Is active is a binary variable with 1 being yes and 0 being no
is_active = np.random.choice([0, 1], size=num_customers)
#Estimated salary is a random float between 10,000 and 100,000
estimated_salary = np.round(np.random.uniform(10000, 100000, size=num_customers), 2)

#Total charges is a function of monthly charges and tenure + noise
total_charges = monthly_charges * tensure + np.random.normal(0, 500, num_customers)
#Ensure total charges are non-negative
total_charges[total_charges < 0] = 0
total_charges = np.round(total_charges, 2)

#Churn is a binary variable with 80% of customers not churning and 20% churning
churn = np.random.choice([0, 1], size=num_customers, p=[0.8, 0.2])


#Create a pandas DataFrame with the generated data
data = pd.DataFrame({
    'CustomerId': customer_id,
    'Gender': gender,
    'Age': age,
    'Geography': geography,
    'MonthlyCharges': monthly_charges,
    'Tenure': tensure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCard': has_card,
    'IsActive': is_active,
    'EstimatedSalary': estimated_salary,
    'TotalCharges': total_charges,
    'Churn': churn
})
#Save the data to a CSV file
data.to_csv('data.csv', index=False)