import streamlit as st
import joblib

# Chargement
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Classification de log")

user_input = st.text_area("Entrez votre texte ici :")

if st.button("Prédire"):
    if user_input.strip() == "":
        st.warning("Veuillez entrer un texte.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)
        st.success(f"Classe prédite : {prediction[0]}")
