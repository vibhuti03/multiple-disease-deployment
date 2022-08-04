#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 16:53:18 2022

@author: apple
"""

import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

diabetes_model = pickle.load(open('/Users/apple/Downloads/Multiple-Disease-Prediction-main/Models/diabetes_model.sav','rb'))

# input_data = (5,166,19,25.8,51)

# # changing the input_data to numpy array
# input_data_as_numpy_array = np.asarray(input_data)

# # reshape the array as we are predicting for one instance
# input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# prediction = diabetes_model.predict(input_data_reshaped)
# print(prediction)

# if (prediction[0] == 0):
#   print('The person is not diabetic')
# else:
#   print('The person is diabetic')

cancer_model = pickle.load(open('/Users/apple/Downloads/Multiple-Disease-Prediction-main/Models/cancer_model.sav','rb'))
heart_model = pickle.load(open('/Users/apple/Downloads/Multiple-Disease-Prediction-main/Models/heart_model.sav','rb'))


def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = diabetes_model.predict(input_data_reshaped)
    print(prediction)
    
    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
    
def cancer_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = cancer_model.predict(input_data_reshaped)
    print(prediction)
    
    if (prediction[0] == 0):
        return 'The person has malignant cancer'
    else:
        return 'The person has benign cancer'
    
def heart_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = heart_model.predict(input_data_reshaped)
    print(prediction)
    
    if (prediction[0] == 0):
        return 'The person does not have any heart related disease'
    else:
        return 'The person is having heart related disease'
    
def main():
    with st.sidebar:
        selected = option_menu('Disease Prediction System', ['Diabetes', 'Cancer', 'Heart'], default_index=0)
    if(selected == 'Diabetes'):
        st.title('Diabetes Prediction System')
    
        Pregnancies = st.text_input('No. of Pregnancies')
        Glucose= st.text_input('Glucose level')
        SkinThickness= st.text_input('Skin Thickness')
        BMI= st.text_input('BMI')
        Age= st.text_input('Age')
        diagnosis = ''

        if st.button('Predict'):
            diagnosis = diabetes_prediction([Pregnancies, Glucose, SkinThickness, BMI, Age])

        st.success(diagnosis)
    
    if(selected == 'Cancer'):
        st.title('Cancer Prediction System')
        
        radius_mean = st.text_input('Mean Radius')
        perimeter_mean= st.text_input('Mean Perimeter')
        area_mean= st.text_input('Mean Area')
        compactness_mean= st.text_input('Mean Compactness')
        concavity_mean= st.text_input('Mean Concativity')
        concave_points_mean= st.text_input('Mean Concave Points')
        radius_se= st.text_input('Standard Error Radius')
        perimeter_se= st.text_input('Standard Error Perimeter')
        area_se= st.text_input('Standard Error Age')
        radius_worst= st.text_input('Worst Radius')
        texture_worst= st.text_input('Worst Texture')
        perimeter_worst= st.text_input('Worst Perimeter')
        area_worst= st.text_input('Worst Area')
        compactness_worst= st.text_input('Worst Compactness')
        concavity_worst= st.text_input('Worst Concavity')
        concave_points_worst= st.text_input('Worst Concave Points')
        diagnosis = ''

        if st.button('Predict'):
            diagnosis = cancer_prediction([radius_mean, perimeter_mean, area_mean,compactness_mean, concavity_mean, concave_points_mean,radius_se, perimeter_se, area_se, radius_worst, texture_worst,perimeter_worst, area_worst, compactness_worst, concavity_worst,concave_points_worst])

        st.success(diagnosis)
        
    if(selected == 'Heart'):
        st.title('Heart Prediction System')
        
        age= st.text_input('Age')
        male= st.text_input('Male(1 or 0)')
        cigsPerDay= st.text_input('Ciggerates per day')
        totChol= st.text_input('Cholestrol')
        bp= st.text_input('BP')
        glucose= st.text_input('Glucose')
        diagnosis = ''

        if st.button('Predict'):
            diagnosis = heart_prediction([age, male, cigsPerDay, totChol, bp, glucose])

        st.success(diagnosis)
    
if __name__ == '__main__':
    main()