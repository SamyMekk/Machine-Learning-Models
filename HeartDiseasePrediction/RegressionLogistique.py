# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:25:47 2023

@author: smekkaoui
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
import matplotlib.pyplot as plt



# Traitement des Données

Dataset=pd.read_csv("data/Cleveland.csv",index_col=0)
Dataset.columns=["Age","Sexe","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]
Data=Dataset.replace('?', np.NaN).copy()
Data["thal"]=Data["thal"].astype("float64")
Data["ca"]=Data["ca"].astype("float64")

# Check des Données
#print(Data.dtypes)

# On supprime les valeurs manquantes

Data=Data.dropna()

# Recodage Variable
def Target(x):
    if x==1 or x==2 or x==3 or x==4:
        return "Oui"
    else:
        return "Non"
    
def sex(x):
    if x==float(0):
        return "Femme"
    else:
        return "Homme"

def cp(x):
    if x==1:
        return "Angine stable"
    elif x==2:
        return "Angine Instable"
    elif x==3:
        return "Autres douleurs"
    else:
        return "Asymptomatique"
    
def fbs(x):
    if x==0:
        return "Non"
    else:
        return "Oui"
    
def restecg(x):
    if x==0:
        return "Normal"
    elif x==1:
        return "Anomalies"
    else:
        return "Hypertrophie"
    
def exang(x):
    if x==0:
        return "Non"
    else:
        return "Oui"
    
def slope(x):
    if x==1:
        return "En Hausse"
    elif x==2:
        return "Stable"
    else:
        return "En Baisse"
    
def ca(x):
    if x==0:
        return "Absence d'Anomalie"
    elif x==1:
        return "Faible"
    elif x==2:
        return "Moyen"
    else:
        return "Elevé"
    
def thal(x):
    if x==float(3):
        return "Non"
    elif x==float(6):
        return "Thalassémie sous contrôle"
    else:
        return "Thalassémie instable"
   
    
# Apply Fonctions Modèles

Data["Sexe"]=Data["Sexe"].apply(sex)
Data["cp"]=Data["cp"].apply(cp)
Data["target"]=Data["target"].apply(Target)
Data["fbs"]=Data["fbs"].apply(fbs)
Data["restecg"]=Data["restecg"].apply(restecg)
Data["exang"]=Data["exang"].apply(exang)
Data["slope"]=Data["slope"].apply(slope)
Data["ca"]=Data["ca"].apply(ca)
Data["thal"]=Data["thal"].apply(thal)



DataTest=Data.copy()
# DataTest

DataClean = pd.get_dummies(DataTest, columns=['Sexe',"cp","fbs","restecg","exang","slope","ca","thal","target"], drop_first=True)
X=DataClean.drop(columns=['target_Oui'])
y=DataClean["target_Oui"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = LogisticRegression(random_state = 0, penalty = 'none')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_train)
# accuracy_score(y_train, y_pred)

# Matrice de Confusion

# ConfusionMatrixDisplay.from_estimator(classifier,X_test,y_test)


# Plot Courbe ROC

# RocCurveDisplay.from_estimator(classifier,X_test,y_test)