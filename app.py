import streamlit as st
import numpy as np
import pandas as pd
import pickle

pipe = pickle.load(open('pipe.pkl','rb'))

st.header('Car Price Predictor')

# year
year = st.number_input('Make year')
# km driven
kms = st.number_input('KMs driven')
# fuel
fuel = st.selectbox('Fuel Type',('Diesel','Petrol'))
# seller type
seller = st.selectbox('Seller Type',('Individual','Dealer'))
# transmission
transmission = st.selectbox('Transmission',('Manual','Automatic'))
# owner
owner = st.selectbox('Owner',('First Owner','Second Owner','Third Owner'))
# mileage
mileage = st.number_input('Mileage')
# engine
engine = st.number_input('Engine')
# max power
power = st.number_input('Max Power')
# seats
seats = st.number_input('Seats')
# brand
brand = st.selectbox('Brand',('Maruti','Hyundai','Mahindra','Tata','Ford','Honda','Toyota','Renault','Chevrolet','Volkswagen'))

if st.button('Predict Price'):
    # form a numpy array(1,11)
    input = np.array([[year, kms, fuel, seller, transmission, owner, mileage, engine, power, seats, brand]])
    input = pd.DataFrame(input,columns=['year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats','brand'])
    #st.dataframe(input)
    y_pred = pipe.predict(input)
    st.title("Rs " + str(np.round(y_pred[0])))