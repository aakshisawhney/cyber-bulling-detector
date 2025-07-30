import streamlit as st
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import nltk
nltk.download('stopwords')

# Load Model and Vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

stop_words = stopwords.words('english')

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def predict(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned]).toarray()
    pred = model.predict(vec)
    return pred[0]

# Streamlit UI
st.title("Cyberbullying Detector 🚫💬")
user_input = st.text_area("Enter a comment to analyze:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        result = predict(user_input)
        st.success(f"Prediction: **{result}**")
