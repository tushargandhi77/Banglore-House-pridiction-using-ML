import streamlit as st
import numpy as np
import pickle
import pandas as pd

pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Banglore House Price Prediction")

area = st.selectbox('Area',df['area_type'].unique())

location = st.selectbox('Location',df['location'].unique())

size = st.selectbox('Size',df['size'].unique())

square_fit = st.number_input('Square_fit')

bath = st.selectbox('Bathroom',df['bath'].unique())

balcony = st.selectbox('Balcony',df['balcony'].unique())

if st.button('Predict Price'):
    query = np.array([area,location,size,square_fit,bath,balcony])
    query = query.reshape(1,6)
    st.title("The predicted Price: " + str(int(pipe.predict(query))))