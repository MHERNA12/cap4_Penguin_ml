# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 16:19:36 2025

@author: Equipo
"""

import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


st.title("Clasificador de pingüinos: Machine Learning App")
st.write("Esta aplicación usa 6 inputs para predecir la especie de pingüino "
         "usando el modelo construido sobre el dataset Palmer Penguins. "
         "Usa el formulario debajo")

penguin_file = st.file_uploader('Carga tus propios datos')

if penguin_file is None:
    penguin_df = pd.read_csv('penguins.csv')
    rf_pickle = open('random_forest_penguin.pickle', 'rb')
    map_pickle = open('output_penguin.pickle', 'rb')
    
    rfc = pickle.load(rf_pickle)
    unique_penguin_mapping = pickle.load(map_pickle)
    rf_pickle.close()
    map_pickle.close()

else:
    penguin_df = pd.read_csv(penguin_file)
    penguin_df.dropna(inplace=True)
    output = penguin_df['species']
    features = penguin_df.loc[:,['island', 'bill_length_mm', 'bill_depth_mm','flipper_length_mm', 'body_mass_g', 'sex']]

    features = pd.get_dummies(features)
    output, uniques = pd.factorize(output)

    x_train, x_test, y_train, y_test = train_test_split(features, output,test_size=.8)

    rfc = RandomForestClassifier(random_state=15)
    rfc.fit(x_train.values, y_train)
    y_pred = rfc.predict(x_test.values)
    score = accuracy_score(y_pred, y_test)
    st.write(f""" Hemos entrenado un modelo Random Forest con estos datos, 
             con un resultado de {score}. Usa los inputs para probar el modelo""")

with st.form('user_inputs'):
    island = st.selectbox("Penguin Island", options=['Biscoe', "Dream", 'Torgerson'])
    
    sex = st.selectbox("Sex", options=['Female', 'Male'])
    
    bill_length = st.number_input('Longitud pico', min_value=0)
    bill_depth = st.number_input('Profundidad pico', min_value=0)
    flipper_length = st.number_input('Longitud aleta', min_value=0)
    body_mass = st.number_input('Masa corporal (g)', min_value=0)
    
    st.form_submit_button('Calcula')
    
    island_biscoe, island_dream, island_torgerson = 0, 0, 0
    if island == 'Biscoe':
        island_biscoe=1
    elif island == 'Dream':
        island_dream = 1
    elif island == 'Torgerson':
        island_torgerson=1
        
    sex_female, sex_male = 0,0
    if sex=='Female':
        sex_female=1
    elif sex=='Male':
        sex_male=1
    
new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length, body_mass, 
                               island_biscoe, island_dream, island_torgerson, 
                               sex_female, sex_male]])

prediction_species = unique_penguin_mapping[new_prediction][0]

st.header(f'Según los parámetros introducidos el pinguino pertenece a la especie {prediction_species}')

user_inputs = [island, sex, bill_length, bill_depth, flipper_length, body_mass]
st.write(f"""Parametros introducidos: {user_inputs}""".format())
st.write("""We used a machine learning (Random Forest)
model to predict the species, the features
used in this prediction are ranked by
relative importance below."""
)
st.image('feature_importance.png')

st.write(
"""Below are the histograms for each
continuous variable separated by penguin
species. The vertical line represents
your the inputted value."""
)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['bill_length_mm'],
hue=penguin_df['species'])
plt.axvline(bill_length)
plt.title('Bill Length by Species')
st.pyplot(ax)
fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['bill_depth_mm'],
hue=penguin_df['species'])
plt.axvline(bill_depth)
plt.title('Bill Depth by Species')
st.pyplot(ax)
fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['flipper_length_mm'],
hue=penguin_df['species'])
plt.axvline(flipper_length)
plt.title('Flipper Length by Species')
st.pyplot(ax)




