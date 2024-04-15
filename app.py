import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib

# Load the SVM model and TfidfVectorizer
svc_model = joblib.load('svm_model.joblib')
with open('tfidf_vectorizer.pickle', 'rb') as handle:
    tfidf_vectorizer = pickle.load(handle)

# Load the Neural Network model and tokenizer
nn_model = load_model('nn_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to preprocess and predict sentiment using SVM
def data_processing(text):
    # Simple example: Remove special characters and convert to lowercase
    processed_text = re.sub(r'[^a-zA-Z\s]', '', text)
    processed_text = processed_text.lower()
    return processed_text

def predict_sentiment_svm():
    review_text = text_review.get("1.0", tk.END)

    if review_text.strip():
        # Preprocess the review
        processed_review = data_processing(review_text)

        # Vectorize the review using TfidfVectorizer
        vectorized_review = tfidf_vectorizer.transform([processed_review])

        # Predict sentiment using the SVM model
        prediction = svc_model.predict(vectorized_review)

        # Display the sentiment to the user
        sentiment = "Positive" if prediction[0] == 'pos' else "Negative"
        messagebox.showinfo("Sentiment Prediction (SVM)", f"Predicted Sentiment: {sentiment}")

    else:
        messagebox.showwarning("Incomplete Submission", "Please enter your review before predicting sentiment.")

# Function to preprocess and predict sentiment using Neural Network
def predict_sentiment_nn():
    review_text = text_review.get("1.0", tk.END)

    if review_text.strip():
        # Preprocess the review
        processed_review = data_processing(review_text)

        # Vectorize the review using the Neural Network tokenizer
        sequence = tokenizer.texts_to_sequences([processed_review])
        sequence = pad_sequences(sequence, maxlen=2114)  

        # Predict sentiment using the Neural Network model
        prediction = nn_model.predict(sequence)

        # Display the sentiment to the user
        sentiment = "Positive" if prediction[0] > 0.5 else "Negative"
        messagebox.showinfo("Sentiment Prediction (NN)", f"Predicted Sentiment: {sentiment}")

    else:
        messagebox.showwarning("Incomplete Submission", "Please enter your review before predicting sentiment.")

# Create the main window
window = tk.Tk()
window.title("Movie Review Sentiment Prediction")

# Create and place widgets
label_review = tk.Label(window, text="Your Review:")
label_review.grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)

text_review = tk.Text(window, height=5, width=40)
text_review.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)

predict_button_svm = tk.Button(window, text="Predict Sentiment (SVM)", command=predict_sentiment_svm)
predict_button_svm.grid(row=1, column=0, pady=10)

predict_button_nn = tk.Button(window, text="Predict Sentiment (NN)", command=predict_sentiment_nn)
predict_button_nn.grid(row=1, column=1, pady=10)

# Start the GUI event loop
window.mainloop()
