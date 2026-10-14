import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# 1. Load and Clean the Dataset
# Ensure 'advertising.csv' is in your working directory
df = pd.read_csv('sales pridiction/backend_ and_MODEL/Advertising.csv')

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# 2. Define Features (X) and Target (y)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# 3. Split and Train the Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Quick evaluation print to verify model health
predictions = model.predict(X_test)
print(f"Model Trained Successfully! (R² Score: {metrics.r2_score(y_test, predictions):.2f})\n")

# 4. Real-Time Interactive Prediction Loop
print("=" * 50)
print(" LIVE SALES PREDICTION SYSTEM ")
print(" Enter your advertising budgets to predict sales output.")
print(" Type -1 for TV budget anytime to exit.")
print("=" * 50)

while True:
    try:
        tv_input = float(input("\nEnter TV advertising budget: "))
        if tv_input == -1:
            print("Exiting prediction system. Good luck with your application!")
            break
            
        radio_input = float(input("Enter Radio advertising budget: "))
        newspaper_input = float(input("Enter Newspaper advertising budget: "))
        
        # Prepare the input data frame matching the model's expected columns
        user_data = pd.DataFrame([[tv_input, radio_input, newspaper_input]], 
                                 columns=['TV', 'Radio', 'Newspaper'])
        
        # Predict the real output
        predicted_sales = model.predict(user_data)[0]
        
        print("-" * 30)
        print(f"👉 Predicted Sales Output: {predicted_sales:.2f} units")
        print("-" * 30)
        
    except ValueError:
        print("Invalid input! Please enter numeric values for the budgets.")