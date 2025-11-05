# lets import the streamlit and other libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# load the model
with open('knn_model.pkl', 'rb') as f:
    knn = pickle.load(f)

st.title("KNN Classifier Web App")
st.write("This is a simple web application to demonstrate KNN Classifier using Streamlit.")
# input features
sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=1.5)
petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=0.2)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = knn.predict(input_data)
    # display the prediction
    # if 0 then setosa, 1 then versicolor, 2 then virginica
    if prediction[0] == 0:
        st.write("The predicted class is: Setosa")
    elif prediction[0] == 1:
        st.write("The predicted class is: Versicolor")
    else:
        st.write("The predicted class is: Virginica")


