import streamlit as st
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load model
model = tf.keras.models.load_model("skin_cancer_metadata_final.h5")

# Create new scaler (use same logic as training)
scaler = StandardScaler()

st.title(" Skin Cancer Prediction (Metadata-Based)")
st.write("Predicts if the lesion is benign or malignant using patient metadata.")

# Collect inputs (example)
age = st.number_input("Age", 0, 120, 45)
smoke = st.selectbox("Smoke", [0, 1])
drink = st.selectbox("Drink", [0, 1])
pesticide = st.selectbox("Pesticide Exposure", [0, 1])
diameter_1 = st.number_input("Diameter 1 (mm)", 0.0, 50.0, 5.0)
diameter_2 = st.number_input("Diameter 2 (mm)", 0.0, 50.0, 4.0)

# Add more fields as needed
input_data = np.array([[age, smoke, drink, 0, 0, pesticide, 0, 0, 0, 1, 1, 3, 0, 0, 0, 0, 0, 0, 1, 0, diameter_1, diameter_2]])

scaled_input = scaler.fit_transform(input_data)  # Use same preprocessing
prediction = model.predict(scaled_input)
result = "Malignant" if prediction[0][0] > 0.5 else " Benign"

st.subheader(result)
