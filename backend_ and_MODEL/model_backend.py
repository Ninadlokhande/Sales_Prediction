import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def load_and_train_model():
    csv_path = 'Advertising.csv'
    if not os.path.exists(csv_path):
        csv_path = '../Advertising.csv' # Fallback path when running pages
        
    df = pd.read_csv(csv_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    r2 = metrics.r2_score(y_test, predictions)
    mae = metrics.mean_absolute_error(y_test, predictions)
    
    return df, model, X_test, y_test, predictions, r2, mae

def predict_sales(model, tv, radio, newspaper):
    input_data = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])
    return model.predict(input_data)[0]