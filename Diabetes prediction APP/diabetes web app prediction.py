# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:00:28 2024

@author: GL RAO
"""

import numpy as np
import pickle
import streamlit as st

loaded_model=pickle.load(open('D:\Machine learning/trained_model.sav','rb'))

def diabetes_prediction(input_data):
    #input_data = (5,166,72,19,175,25.8,0.587,51)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # standardize the input data
    #std_data = scaler.transform(input_data_reshaped)
    #print(std_data)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
def main():
    st.title("Diabetes Prediction Web App")
    
    #Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    
    
    Pregnancies=st.text_input("Number of pregnancies")
    Glucose=st.text_input("Glucose level")
    BloodPressure=st.text_input("Bp")
    SkinThickness=st.text_input("Skin thickness")
    Insulin=st.text_input("Insulin")
    BMI=st.text_input("Bmi")
    DiabetesPedigreeFunction=st.text_input("DiabetesPedigreeFunction")
    Age=st.text_input("age")
    
    diagnosis=''
    
    if st.button("Diabetes Test Result"):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
    
    
if __name__== '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    