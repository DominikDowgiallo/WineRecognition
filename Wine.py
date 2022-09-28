import streamlit as st
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler




st.write("""
    This app predicts class of wine from wine dataset from sklearn library
    """)



# loading data

wine = load_wine()
X = pd.DataFrame(wine.data, columns= wine.feature_names)
Y = wine.target


dict = {}
st.sidebar.header('Parameters')

for i in wine.feature_names:
    dict[i] = []
def user_input_features():
    for i in wine.feature_names:
        globals()[f'x{i}'] = st.sidebar.slider(i, float(X[i].min()), float(X[i].max()), float(X[i].mean()))
        dict[i] = globals()[f'x{i}']
    user_values = pd.DataFrame(dict, index = [0])
    return user_values


df = user_input_features()

st.header('Input parameters')
st.write(df)
# standarizing data
scaler = StandardScaler().fit(X)
X = pd.DataFrame(scaler.transform(X))
df = pd.DataFrame(scaler.transform(df))

# Building ML Model
knn = KNeighborsClassifier()
knn.fit(X,Y)

prediction = knn.predict(df)
prediction_proba = knn.predict_proba(df)


st.header('Prediction')
st.write(prediction)
st.header('Probablity')
st.write(prediction_proba)
