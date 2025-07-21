import streamlit as st
import pandas as pd 
import numpy as np
from PREDICT import predict


st.title('Classifying Iris Flowers')


st,header("plant features")

col1, col2 = st.columns(2)

with col1:
    st.text('Sepal Characteristics')
    sepal_l=st.slider('Sepal Length (cm)',1.0, 8.0, 0.5)
    sepal_w=st.slider('Sepal Width (cm)',3.0, 4.4, 0.5)

with col1:
    st.text('Sepal Characteristics')
    petal_l=st.slider('Petal Length (cm)',1.0, 7.0, 0.5)
    petal_w=st.slider('Petal Width (cm)',0.1, 2.5, 0.5)

st.text('')
if st.button("Predict Type of Iris"):
    result=np.array(np.array([[sepal_l, sepal_w,petal_l,petal_w]]))
    predicted_class = result[0]
    st.success(f"Predicted:{predicted_class.title()}")

    image_path = f"images/{predicted_class.lower()}.jpg"
    st.image(image_path, caption= predicted_class.title(),use_container_width=True)