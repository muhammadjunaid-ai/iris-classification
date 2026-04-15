import streamlit as st
import joblib
import numpy as np

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🌸 Iris Flower Classifier")

sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

if st.button("Predict"):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    data = scaler.transform(data)

    prediction = model.predict(data)

    species = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"Predicted: {species[prediction[0]]}")