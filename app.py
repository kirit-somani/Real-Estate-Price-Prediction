from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer

app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def predict_page():
    min_area = int(df['Total_Area'].min())
    max_area = int(df['Total_Area'].max())
    avg_area = int(df['Total_Area'].mean())
    min_baths = int(df['Baths'].min())
    max_baths = int(df['Baths'].max())
    min_bhk = int(df['Number of BHK'].min())
    max_bhk = int(df['Number of BHK'].max())
    
    return render_template('index.html', 
                         cities=cities,
                         locations=locations,
                         min_area=min_area,
                         max_area=max_area,
                         avg_area=avg_area,
                         min_baths=min_baths,
                         max_baths=max_baths,
                         min_bhk=min_bhk,
                         max_bhk=max_bhk)


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
    
    # Convert numeric columns
    df['Total_Area'] = pd.to_numeric(df['Total_Area'], errors='coerce')
    df['Price_per_SQFT'] = pd.to_numeric(df['Price_per_SQFT'], errors='coerce')
    df['Baths'] = pd.to_numeric(df['Baths'], errors='coerce')
    df['Number of BHK'] = pd.to_numeric(df['Number of BHK'], errors='coerce')
    
    # Remove outliers
    for col in ['Price', 'Total_Area', 'Price_per_SQFT']:
        df = df[df[col].notna()]
        low, high = df[col].quantile([0.01, 0.99])
        df = df[df[col].between(low, high)]
    
    # Clean BHK data - typically houses have 1-6 BHK
    df = df[df['Number of BHK'].between(1, 6)]
    
    # Select relevant features
    features = ['Total_Area', 'Baths', 'Number of BHK', 'Price_per_SQFT', 'City', 'Location']
    target = 'Price'
    
    # Keep only necessary columns
    df = df[[target] + features].copy()
    
    # Handle missing values
    df = df.dropna()
    
    return df

# Load and process data
df = wrangle('Pricing.csv')

# Get feature lists
features = ['Total_Area', 'Baths', 'Number of BHK', 'Price_per_SQFT', 'City', 'Location']
target = 'Price'

# Split data
X_train = df[features]
y_train = df[target]

# Calculate baseline metrics
baseline_pred = np.full_like(y_train, y_train.mean())
baseline_mae = np.mean(np.abs(baseline_pred - y_train))

# Create and train model pipeline
model = make_pipeline(
    OneHotEncoder(use_cat_names=True, handle_unknown='ignore'),
    StandardScaler(),
    Ridge(alpha=1.0)
)

# Fit the model
model.fit(X_train, y_train)

# Calculate Mean Absolute Error
from sklearn.metrics import mean_absolute_error
y_pred = model.predict(X_train)
mae = mean_absolute_error(y_train, y_pred)

# Calculate baseline MAE (using mean prediction)
baseline_pred = [y_train.mean()] * len(y_train)
baseline_mae = mean_absolute_error(y_train, baseline_pred)

# Calculate improvement over baseline
improvement = ((baseline_mae - mae) / baseline_mae) * 100

print(f"\nModel Performance:")
print(f"Mean property price: Rs. {y_train.mean():,.2f}")
print(f"Baseline MAE: Rs. {baseline_mae:,.2f}")
print(f"Training MAE: Rs. {mae:,.2f}")
print(f"Improvement over baseline: {improvement:.1f}%")

# Get unique values for dropdowns
cities = sorted(df['City'].unique())
locations = sorted(df['Location'].unique())

def make_prediction(area, price_per_sqft, baths, bhk, city, location):
    # Create DataFrame with features in the exact same order as training
    df_pred = pd.DataFrame([
        {
            'Total_Area': float(area),
            'Baths': int(baths),
            'Number of BHK': int(bhk),
            'Price_per_SQFT': float(price_per_sqft),
            'City': city,
            'Location': location
        }
    ])
    # Ensure columns are in the same order as training
    df_pred = df_pred[features]
    prediction = model.predict(df_pred)[0]
    return f"₹{prediction:,.2f}"



@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate required fields
        required_fields = ['area', 'baths', 'bhk', 'city', 'location', 'price_per_sqft']
        for field in required_fields:
            if field not in request.form:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400
            if not request.form[field]:
                return jsonify({'success': False, 'error': f'Field cannot be empty: {field}'}), 400

        # Convert and validate numeric fields
        try:
            area = float(request.form['area'])
            baths = int(request.form['baths'])
            bhk = int(request.form['bhk'])
            price_per_sqft = float(request.form['price_per_sqft'])
        except ValueError:
            return jsonify({'success': False, 'error': 'Invalid numeric values provided'}), 400

        # Validate ranges
        if not (1 <= bhk <= 6):
            return jsonify({'success': False, 'error': 'BHK must be between 1 and 6'}), 400
        if not (1000 <= price_per_sqft <= 60000):
            return jsonify({'success': False, 'error': 'Price per sq ft must be between 1000 and 20000'}), 400

        # Validate city and location
        if request.form['city'] not in cities:
            return jsonify({'success': False, 'error': 'Invalid city selected'}), 400
        if request.form['location'] not in locations:
            return jsonify({'success': False, 'error': 'Invalid location selected'}), 400
        
        # Make prediction
        prediction = make_prediction(area, price_per_sqft, baths, bhk, request.form['city'], request.form['location'])
        return jsonify({'success': True, 'prediction': prediction})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_locations/<city>')
def get_locations(city):
    city_locations = sorted(df[df['City'] == city]['Location'].unique())
    return jsonify(city_locations)

if __name__ == '__main__':
    app.run(debug=True, port=5001)