import streamlit as st
import pandas as pd
import joblib

# Load the trained model and feature list
model = joblib.load('models/final_model.pkl')
features = joblib.load('feature_list.pkl')


# Define the input fields
st.title('Price Prediction using Gradient Boosting Regressor')

# Define categorical options
brand_options = ['apple', 'boat', 'dizo', 'fire-boltt', 'garmin', 'noise', 'pebble', 'samsung', 'zebronics']
dial_shape_options = ['Contemporary', 'Curved', 'Oval', 'Rectangle', 'Square']
strap_color_options = ['Blue', 'Green', 'Grey', 'Others', 'Pink', 'Red', 'Silver']
strap_material_options = ['Fluoroelastomer', 'Leather', 'Other', 'Others', 'Rubber', 'Silicon', 'Stainless Steel']

# Initialize input data dictionary
input_data = {feature: 0 for feature in features}

# Create input fields for numerical features
current_price = st.number_input('Enter Current Price', value=0.0)
original_price = st.number_input('Enter Original Price', value=0.0)
rating = st.number_input('Enter Rating', value=0.0)
number_of_ratings = st.number_input('Enter Number of Ratings', value=0)
battery_life_days = st.number_input('Enter Battery Life (Days)', value=0.0)

# Update input data dictionary
input_data['Current Price'] = current_price
input_data['Original Price'] = original_price
input_data['Rating'] = rating
input_data['Number of Ratings'] = number_of_ratings
input_data['Battery Life (Days)'] = battery_life_days

# Create dropdowns for categorical features
selected_brand = st.selectbox('Select Brand', brand_options)
selected_dial_shape = st.selectbox('Select Dial Shape', dial_shape_options)
selected_strap_color = st.selectbox('Select Strap Color', strap_color_options)
selected_strap_material = st.selectbox('Select Strap Material', strap_material_options)

# Set one-hot encoded features based on selections
for brand in brand_options:
    input_data[f'Brand_{brand}'] = 1 if brand == selected_brand else 0

for shape in dial_shape_options:
    input_data[f'Dial Shape_{shape}'] = 1 if shape == selected_dial_shape else 0

for color in strap_color_options:
    input_data[f'Strap Color_{color}'] = 1 if color == selected_strap_color else 0

for material in strap_material_options:
    input_data[f'Strap Material_{material}'] = 1 if material == selected_strap_material else 0

# Convert input data to DataFrame and ensure correct feature order
input_df = pd.DataFrame([input_data], columns=features)

# Predict the price
if st.button('Predict'):
    prediction = model.predict(input_df).round(2)
    st.write(f'Predicted Price: {prediction[0]}')

# Display the input data
st.write("### Input Data")
st.write(input_df)