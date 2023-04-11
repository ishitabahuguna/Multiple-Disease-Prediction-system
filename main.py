import numpy as np
import pandas as pd
pip install sklearn
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
data1=pd.read_csv('https://github.com/ishitabahuguna/Multiple-Disease-Prediction-system/blob/main/diabetes.csv')
data1.head()
import pickle
import streamlit as st

# loading models
diabetesmodel = pickle.load(open("diabetes.pkl", 'rb'))


heartdiseasemodel = pickle.load(open("C:/Users/Dell/Desktop/Multiple Prediction system/heartmodel.pkl", 'rb'))

breastcancermodel = pickle.load(open("C:/Users/Dell/Desktop/Multiple Prediction system/breast.pkl", 'rb'))


# sidebar customization for navigation purpose

with st.sidebar:
    selected = option_menu('MULTIPLE DISEASE PREDICTION TECHNIQUE',
                           ['DIABETES PREDICTION', 'HEART DISEASE PREDICTION', 'TYPE OF BREAST CANCER PREDICTION'
                            ],
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Diabetes Prediction Page
if (selected == "DIABETES PREDICTION"):

    # page title
    st.title="DIABETES PREDECTION USING :blue[SVM]"
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):
        diab_prediction = diabetesmodel.predict(
            [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        if (diab_prediction[0] == 1):
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if (selected == 'HEART DISEASE PREDICTION'):

    # page title
    st.title='Heart Disease Prediction using Logistic Regression'

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):
        heart_prediction = heartdiseasemodel.predict(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        if (heart_prediction[0] == 0):
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

if (selected == 'TYPE OF BREAST CANCER PREDICTION'):
    # title setting
    st.title=("Type OF Breast Cancer Prediction using Random Forest")
    col1, col2, col3 = st.columns(3)
    with col1:
        radius_mean = st.text_input("Enter Radius Mean")
    with col2:
        texture_mean = st.text_input("Enter Texture Mean")
    with col3:
        perimeter_mean = st.text_input("Enter Perimeter Mean")
    with col1:
        area_mean = st.text_input("Enter Area Mean")
    with col2:
        smoothness_mean = st.text_input("Enter Smoothness Mean")
    with col3:
        concavity_mean = st.text_input("Enter Concavity Mean")
    with col1:
        concavepoints_mean = st.text_input("Enter Concave Points Mean")
    with col2:
        symmetry_mean = st.text_input("Enter Symmetry Mean")
    with col3:
        fractal_dimension_mean = st.text_input("Enter Fractal dimension Mean")

    # code for Prediction
    breast_diagnosis = ''
    # creating a button for Prediction

    if st.button('Breast Cancer Type Test Result'):
        cancer_prediction = breastcancermodel.predict([[radius_mean, texture_mean, perimeter_mean, area_mean,
                                                        smoothness_mean, concavity_mean, concavepoints_mean,
                                                        symmetry_mean, fractal_dimension_mean]])

        if (cancer_prediction[0] == 1):
            breast_diagnosis = "The Pateint is suffering from Malignant Tumor(which can spread throughout the whole body via blood)"
        else:
            breast_diagnosis = "The Patient is suffering from Benign Tumor (which spreads locally)"

    st.success(breast_diagnosis)
