import streamlit as st
import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor

st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Application de prédiction des prix de l'immobilier au Maroc
Cette application mine les données sur les sites d'annonces et prédit en utilisant le *Machine Learning* le prix 
d'une maison ou appartement en se basant sur les charactéristiques propore à chanque 
bien dans plusieurs villes du Maroc. 

**NB**: *Cette application n'a aucune fin commerciale et a été développée à des fins pédagogiques seulement. On prétend nullement à travers cette dernière de pouvoir donner des éstimations d'une grande fiabilité vu la petite taille de l'échantillon (4064). Toutefois, l'algorithme nous permet de prédire sur le test set (Données non consultées par le modèle) à hauteur des 75'%' de fiabilité avec une Cross Validation de 10.*

**Pour une meilleur visualisation des données, un dashboard consultable de préférence sur un ordinateur est disponible sur ce [lien](https://public.tableau.com/app/profile/salah7781/viz/Analyse_March_Immobilier_MAROC/Dashboard1)**
""")
st.write('---')

# Load dataset
dataset = pd.read_csv("C:/Users/hp/housepredictionapp/DATA.csv")
# Set X and y
X = dataset.drop(["Unnamed: 0", "Prix", "Prix / m²"], axis=1)
X_copy = X
y = dataset.Prix

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Merci de remplir les charactéristiques du bien que vous souhaitez éstimer.')

# Features cols
features_cols = X.columns
data = {}

with st.sidebar.form(key ='Form1'):
    Ville = st.selectbox("Ville", (X["Ville"].unique()))
    data["Ville"] = Ville

    Secteur = st.selectbox("Secteur (Choisir 'Autre Secteur' si pas existant)", (X["Secteur"].unique()))
    data["Secteur"] = Secteur

    Type = st.selectbox("Type", (X["Type"].unique()))
    data["Type"] = Type

    SA = st.slider("Surface habitable m²", int(X["Surface habitable m²"].min()),
                     int(X["Surface habitable m²"].max()), int(X["Surface habitable m²"].mean()))
    data["Surface habitable m²"] = SA

    ST = st.slider("Surface totale m²", int(X["Surface totale m²"].min()), 
                    int(X["Surface totale m²"].max()), int(X["Surface totale m²"].mean()))
    data["Surface totale m²"] = ST

    Age = st.selectbox("Âge du bien", (X["Âge du bien"].unique()))
    data["Âge du bien"] = Age

    Salons = st.selectbox("Salons", (X["Salons"].unique()))
    data["Salons"] = Salons

    Chambres = st.selectbox("Chambres", (X["Chambres"].unique()))
    data["Chambres"] = Chambres

    SDB = st.selectbox("Salle de bain", (X["Salle de bain"].unique()))
    data["Salle de bain"] = SDB

    Garage = st.radio("Garage", (X["Garage"].unique()))
    data["Garage"] = Garage

    Jardin = st.radio("Jardin", (X["Jardin"].unique()))
    data["Jardin"] = Jardin

    Piscine = st.radio("Piscine", (X["Piscine"].unique()))
    data["Piscine"] = Piscine

    Balcon = st.radio("Balcon", (X["Balcon"].unique()))
    data["Balcon"] = Balcon

    submitted = st.form_submit_button(label = 'VALIDER')

# features = pd.DataFrame(data, index=[0])
df = pd.DataFrame(data, index=[0])

# Add progress bar

if submitted:
    st.header('Caractéristiques du bien immobilier choisi:')
    st.write(df)
    st.write('---')

    my_bar = st.progress(0)

    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)
    
    # Build Regression Model
    features_to_encode = X.columns[X.dtypes==object].tolist()
    # Encode train set
    X = pd.get_dummies(X, columns = features_to_encode)

    model = RandomForestRegressor(random_state = 25)
    model.fit(X, y)

    # Apply Model to Make Prediction
    X_input = pd.get_dummies(df, columns = features_to_encode)

    # Get missing columns in the training test
    missing_cols = set( X.columns ) - set( X_input.columns )
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        X_input[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    test = X_input[X.columns]

    prediction = model.predict(test)

    # st.header('Prix Prédit (en DHS):')
    st.header(' Prix Estimé = {0:,.2f} DHS'.format(int(prediction)))
    st.write('---')
    # Plot predicted price vs median price in the city
    median_price = dataset[dataset.Ville == data["Ville"]].groupby("Secteur")["Prix"].median().sort_values(ascending=False)[:10]
    st.write('TOP 10 des secteurs par Prix Médian')
    # st.write(median_price)
    st.bar_chart(median_price)
    st.write('---')
