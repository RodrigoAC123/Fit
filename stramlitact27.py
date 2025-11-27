import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


st.write(''' # ¿Eres FIT? ''')
st.image("FIT.jpg", caption="Ser fit toma en cuenta multiples variables para clasificar si realmente lo eres, o no.")

st.header('Datos de evaluación')

def user_input_features():
  # Entrada
  age = st.number_input('Escribe tu edad:', min_value=1, max_value=100, value = 0, step = 1)
  height_cm = st.number_input('Escribe tu altura en cm:', min_value=0, max_value=240, value = 0, step = 1)
  weight_kg = st.number_input('Escribe tu peso en kg:', min_value=0, max_value=200, value = 0, step = 1)
  heart_rate = st.number_input('Escribe tu frecuencia cardiaca promedio en reposo (la media esta entre 70-90):',min_value=0, max_value=180, value = 0, step = 1)
  blood_pressure = st.number_input('Escribe tu presión arterial (la media esta entre = 80-140) :', min_value=0, max_value=180, value = 0, step = 1)
  nutrition_quality = st.number_input('Que tan bien comes del 1 al 10:', min_value=1, max_value=10, value = 0, step = 1)
  activity_index = st.number_input('Que tanto entrenas del 1 al 5:', min_value=1, max_value=5, value = 0, step = 1)
  smokes = st.number_input('Fumas (2 =  NO,  3 = SI):', min_value=2, max_value=3, value = 0, step = 1)
  gender = st.number_input('Genero (Mujer = 0,  Hombre = 1):', min_value=0, max_value=1, value = 0, step = 1)
  

  user_input_data = {'age': age,
                     'height_cm': height_cm,
                     'weight_kg': weight_kg,
                     'heart_rate': heart_rate,
                     'blood_pressure': blood_pressure,
                     'nutrition_quality': nutrition_quality,
                     'activity_index': activity_index
                     'smokes': smokes
                     'gender': gender
                     }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

fit =  pd.read_csv('fit.csv', encoding='latin-1')
X = fit.drop(columns='is_fit')
Y = fit['is_fit']

classifier = DecisionTreeClassifier(max_depth=6, criterion='entropy', min_samples_leaf=50, max_features=6, random_state=1613726)
classifier.fit(X, Y)

prediction = classifier.predict(df)

st.subheader('Predicción')
if prediction == 0:
  st.write('No estas en forma')
elif prediction == 1:
  st.write('Estas en forma')
else:
  st.write('Sin predicción')
