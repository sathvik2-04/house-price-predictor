import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

model = pickle.load(open('house_price.pkl', 'rb'))

scaler = StandardScaler()

st.title("House Price Prediction App")

square_footage = st.number_input('Square Footage', min_value=500, max_value=10000, value=2000)
num_bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=10, value=3)
num_bathrooms = st.number_input('Number of Bathrooms', min_value=1, max_value=10, value=2)
year_built = st.number_input('Year Built', min_value=1900, max_value=2024, value=2000)
lot_size = st.number_input('Lot Size', min_value=0.1, max_value=10.0, value=3.5)
garage_size = st.number_input('Garage Size', min_value=0, max_value=5, value=2)
neighborhood_quality = st.slider('Neighborhood Quality', min_value=1, max_value=10, value=5)

# Create dataframe
input_features = pd.DataFrame({
    'Square_Footage': [square_footage],
    'Num_Bedrooms': [num_bedrooms],
    'Num_Bathrooms': [num_bathrooms],
    'Year_Built': [year_built],
    'Lot_Size': [lot_size],
    'Garage_Size': [garage_size],
    'Neighborhood_Quality': [neighborhood_quality]
})

# Scale the features
input_features_scaled = input_features.copy()
input_features_scaled[['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality']] = scaler.fit_transform(input_features)

if st.button('Predict Price'):
    predictions = model.predict(input_features_scaled)
    output = round(predictions[0], 2)
    st.success(f'Price Prediction: ${output:,.2f}')