import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from category_encoders import OneHotEncoder

def wrangle(filepath):
    """Process real estate data for model training."""
    # Read CSV file
    df = pd.read_csv(filepath)
    
    # Clean price column
    def convert_price(price):
        try:
            # Remove currency symbol and commas
            price = str(price).replace('\u20b9', '').replace(',', '')
            
            # Handle lakhs
            if 'L' in price:
                price = float(price.replace(' L', '')) * 100000
            # Handle crores
            elif 'Cr' in price:
                price = float(price.replace(' Cr', '')) * 10000000
            # Handle acs (assume it means acres and convert to lakhs)
            elif 'acs' in price:
                price = float(price.replace('acs', '')) * 100000
            else:
                price = float(price)
            return price
        except:
            return np.nan
    
    df['Price'] = df['Price'].apply(convert_price)
    
    # Convert numeric columns
    df['Total_Area'] = pd.to_numeric(df['Total_Area'], errors='coerce')
    df['Baths'] = pd.to_numeric(df['Baths'], errors='coerce')
    df['Number of BHK'] = pd.to_numeric(df['Number of BHK'], errors='coerce')
    
    # Remove outliers
    for col in ['Price', 'Total_Area']:
        df = df[df[col].notna()]
        low, high = df[col].quantile([0.01, 0.99])
        df = df[df[col].between(low, high)]
    
    return df

def create_visualizations(df):
    """Create and save analysis plots"""
    plt.style.use('default')
    
    # Create multiple plots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Area vs Price scatter plot with regression line
    plt.subplot(3, 2, 1)
    sns.regplot(data=df, x='Total_Area', y='Price', scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
    plt.title('Area vs Price with Regression Line', fontsize=12)
    plt.xlabel('Total Area (sq ft)')
    plt.ylabel('Price (Rs)')
    plt.show()
   
    # Train model for residual plot
    features = ['Total_Area', 'Baths', 'City', 'Location']
    X = df[features]
    y = df['Price']
    
    model = make_pipeline(
        OneHotEncoder(use_cat_names=True, handle_unknown='ignore'),
        SimpleImputer(strategy='median'),
        StandardScaler(),
        Ridge(alpha=1.0)
    )
    
    model.fit(X, y)
    y_pred = model.predict(X)
    
    plt.subplot(3, 2, 2)
    sns.residplot(x=y_pred, y=y, lowess=True, line_kws={'color': 'red'})
    plt.title('Residual Plot', fontsize=12)
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Load and process data
    df = wrangle('Pricing.csv')
    
    # Create visualizations
    create_visualizations(df)
    
