import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model and the scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler_x.pkl', 'rb') as scaler_x_file:
    scaler_x = pickle.load(scaler_x_file)

with open('scaler_y.pkl', 'rb') as scaler_y_file:
    scaler_y = pickle.load(scaler_y_file)

# Streamlit app
st.title("Prediksi Kasus DBD")

st.write("""
### Prediksi Kasus DBD dalam satu bulan dengan fitur Curah Hujan (mm) dan Jumlah Hari Hujan (hari) dalam satu bulan.
""")

# User input
curah_hujan = st.number_input("Curah Hujan (mm)", min_value=0.0, step=0.1)
jumlah_hari_hujan = st.number_input("Jumlah Hari Hujan (Hari)", min_value=0.0, step=1.0, max_value=31.0)

# Predict button
if st.button("Predict"):
    # Prepare input data
    input_data = np.array([curah_hujan, jumlah_hari_hujan]).reshape(1, -1)
    input_df = pd.DataFrame(input_data, columns=['Curah Hujan (mm)', 'Jumlah Hari Hujan (hari)'])
    scaled_input = scaler_x.transform(input_data)
    
    # # # Make prediction
    scaled_output = model.predict(scaled_input)
    
  #   # # Ensure the output is 2D before inverse transforming
    scaled_output_reshaped = np.array(scaled_output).reshape(-1, 1)
    output = scaler_y.inverse_transform(scaled_output_reshaped)
    print(output)

    st.subheader(f"Predicted Kasus DBD: {output[0][0]:.2f} Kasus")
