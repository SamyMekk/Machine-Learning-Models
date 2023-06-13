# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:36:10 2023

@author: smekkaoui
"""


import streamlit as st
import requests
from io import BytesIO
import pandas as pd
from RegressionLogistique import *
from PIL import Image


st.title("Modèle de Régression Logistique pour la prédiction de Heart Disease pour la population de Cleveland")

st.header("Voici à quoi ressemble les Données sur lesquels notre étude va se baser ")



def user_input():
    age=st.sidebar.number_input("Choississez l'âge de l'individu ",value= 40)
    sexe=st.sidebar.selectbox("Choississez le sexe de l'individu",('Homme','Femme'))
    cp=st.sidebar.selectbox("Choississez le type d'angine",("Angine stable","Angine Instable","Autres douleurs","Asymptomatique"))
    trestbps=st.sidebar.number_input("Choississez la Tension Artérielle au repos de l'individu",value=120)
    chol=st.sidebar.number_input("Choississez le niveau de cholestérol",value=230)
    fbs=st.sidebar.selectbox("L'individu a-t-il un taux de glycémie à jeun",('Oui','Non'))
    restecg=st.sidebar.selectbox("Choississez le niveau de mesure d'électrocardiographie au repos",("Normal","Hypertrophie","Anomalies"))
    thalach=st.sidebar.number_input("Choissisez la fréquence cardiaque maximale atteinte",value=150)
    exang=st.sidebar.selectbox("Choissisez si l'exercice induit une angine",("Oui","Non"))
    oldpeak=st.sidebar.number_input("Choississez la valeur d'oldpeak",value=0.5)
    slope=st.sidebar.selectbox("Choissisez le risque lié à la slope",("En Hausse","Stable","En Baisse"))
    ca=st.sidebar.selectbox("Choississez le risque lié aux problèmes cardiovasculaires",("Absence d'Anomalie","Faible","Moyen","Elevé"))
    thal=st.sidebar.selectbox("Choissisez le risque lié à la Thalassémie",("Non","Thalassémie sous contrôle","Thalassémie instable"))
    data={"Age":age,
        "Sexe":sexe,
        "cp":cp,
        "trestbps":trestbps,
        "chol":chol,
        "fbs":fbs,
        "restecg":restecg,
        "thalach":thalach,
        "exang":exang,
        "oldpeak":oldpeak,
        "slope":slope,
        "ca":ca,
        "thal":thal,}
    Parametres=pd.DataFrame(data,index=["Caractéristiques"])
    return Parametres
    


st.header("Voici les résultats de la matrice de confusion :")


url_icon = "https://raw.githubusercontent.com/SamyMekk/Machine-Learning-Models/main/HeartDiseasePrediction/MatrixConfusion.png"
response = requests.get(url_icon)
img = Image.open(BytesIO(response.content))

st.image(img, caption='Confusion Matrix')


st.header("Voici les résultats pour la courbe ROC :")

image = Image.open("ROCCurve.png")

st.image(image, caption='Courbe ROC')
# Modèle Prédiction


df=user_input()

st.subheader("Voici les caractéristiques choisis de l'individu : ")

st.dataframe(df.transpose())

DataforCalcul=DataTest.iloc[0:,0:len(DataTest.transpose())-1]

DataForTest=pd.concat([DataforCalcul,df])

Input=pd.get_dummies(DataForTest, columns=['Sexe',"cp","fbs","restecg","exang","slope","ca","thal"], drop_first=True)


Candidat=pd.DataFrame(Input.iloc[-1,:]).transpose()

st.header("Résultat de la prédiction par régression logistique : ")
Prediction={"Prédiction":[np.round(classifier.predict_proba(Candidat)[0][0],4)]}
Resultat=pd.DataFrame.from_dict(data=Prediction)
Resultat.index=["Probabilité de ne pas subir d'attaque cardiaque :"]
st.dataframe(Resultat)