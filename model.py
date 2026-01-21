import pandas as pd
import seaborn as sns
import numpy as np
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def wrangle(filepath):
    """Process real estate data for model training."""
    # Read CSV file
    df = pd.read_csv(filepath)
    
    # Clean price column
    def convert_price(price):
        try:
            # Remove currency symbol and commas
            price = str(price).replace('₹', '').replace(',', '')
            
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
    
    # Convert area to numeric
    df['Total_Area'] = pd.to_numeric(df['Total_Area'], errors='coerce')
    df['Price_per_SQFT'] = pd.to_numeric(df['Price_per_SQFT'], errors='coerce')
    
    # Remove outliers
    for col in ['Price', 'Total_Area', 'Price_per_SQFT']:
        df = df[df[col].notna()]
        low, high = df[col].quantile([0.01, 0.99])
        df = df[df[col].between(low, high)]
    
    # Select relevant features
    features = ['Total_Area', 'Price_per_SQFT', 'Baths', 'Number of BHK', 'City', 'Location']
    target = 'Price'
    
    # Keep only necessary columns
    df = df[[target] + features].copy()
    
    # Handle missing values
    df = df.dropna()
    
    return df

# Load and process data
df = wrangle('Pricing.csv')

# Basic data info
print("Dataset Info:")
print(df.info())

# Feature selection
features = ['Total_Area', 'Price_per_SQFT', 'Baths', 'Number of BHK', 'City', 'Location']
target = 'Price'

# Split data
X_train = df[features]
y_train = df[target]

# Calculate baseline metrics
y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)
baseline_mae = mean_absolute_error(y_train, y_pred_baseline)

# Create and train model pipeline
model = make_pipeline(
    OneHotEncoder(use_cat_names=True, handle_unknown='ignore'),
    SimpleImputer(strategy='median'),
    StandardScaler(),
    Ridge(alpha=1.0)
)

# Fit model and calculate metrics
model.fit(X_train, y_train)
y_pred_training = model.predict(X_train)
training_mae = mean_absolute_error(y_train, y_pred_training)

# Print metrics
print(f"\nModel Performance:")
print(f"Mean property price: Rs. {y_mean:,.2f}")
print(f"Baseline MAE: Rs. {baseline_mae:,.2f}")
print(f"Training MAE: Rs. {training_mae:,.2f}")
print(f"Improvement over baseline: {((baseline_mae - training_mae) / baseline_mae * 100):.1f}%")

# Plot correlation heatmap for numeric features
print("\nFeature Correlations:")
numeric_cols = ['Total_Area', 'Price_per_SQFT', 'Baths', 'Number of BHK', 'Price']
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')

def make_prediction(area, price_per_sqft, baths, bhk, city, location):
    """Make a price prediction for a property."""
    data = {
        'Total_Area': area,
        'Price_per_SQFT': price_per_sqft,
        'Baths': baths,
        'Number of BHK': bhk,
        'City': city,
        'Location': location
    }
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(df)[0]
    return f"Predicted property price: Rs. {prediction:,.2f}"

# Create interactive widget
from ipywidgets import interact, IntSlider, FloatSlider, Dropdown

interact(
    make_prediction,
    area=IntSlider(
        min=int(X_train['Total_Area'].min()),
        max=int(X_train['Total_Area'].max()),
        value=int(X_train['Total_Area'].mean()),
        description='Area (sq ft)'
    ),
    price_per_sqft=IntSlider(
        min=int(X_train['Price_per_SQFT'].min()),
        max=int(X_train['Price_per_SQFT'].max()),
        value=int(X_train['Price_per_SQFT'].mean()),
        description='Price/sqft'
    ),
    baths=IntSlider(
        min=int(X_train['Baths'].min()),
        max=int(X_train['Baths'].max()),
        value=2,
        description='Bathrooms'
    ),
    bhk=IntSlider(
        min=int(X_train['Number of BHK'].min()),
        max=int(X_train['Number of BHK'].max()),
        value=2,
        description='Bedrooms'
    ),
    city=Dropdown(
        options=sorted(X_train['City'].unique()),
        description='City'
    ),
    location=Dropdown(
        options=sorted(X_train['Location'].unique()),
        description='Location'
    )
)

# Print feature importance (for numeric features)
print("\nFeature Importance:")
numeric_features = ['Total_Area', 'Price_per_SQFT', 'Baths', 'Number of BHK']
X_numeric = X_train[numeric_features]
X_scaled = StandardScaler().fit_transform(X_numeric)
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y_train)

importance = pd.DataFrame({
    'Feature': numeric_features,
    'Importance': np.abs(ridge.coef_)
})
importance = importance.sort_values('Importance', ascending=False)
print(importance)