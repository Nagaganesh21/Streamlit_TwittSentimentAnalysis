from flask import Flask, render_template, request
import pickle
from pycaret.regression import *
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import config
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

# Load the TF-IDF vectorizer
tfidf = pickle.load(open("tfidf_vectorizer.pkl", 'rb'))

# Load the model
model = pickle.load(open("lr_model.pkl", 'rb'))

# Create Flask app
app = Flask(__name__)

# Define home page
@app.route('/')
def Home():
    return render_template("index.html")

# Define predict method
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        # Preprocess the input message (e.g., remove stopwords, lemmatize)
        message = preprocess_message(message)
        data = [message]
        vect = tfidf.transform(data)  # No need to convert to array
        my_prediction = model.predict(vect)
    return render_template('index.html', prediction_text=f'Prediction is: {"Positive Tweet" if my_prediction == 1 else "Negative Tweet"}')

def preprocess_message(message):
    # Your preprocessing steps here (e.g., removing stopwords, lemmatization)
    # For example:
    # Stopwords removal
    stop_words = set(stopwords.words('english'))
    message = ' '.join([word for word in message.split() if word.lower() not in stop_words])
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    message = ' '.join([lemmatizer.lemmatize(word) for word in message.split()])
    return message

if __name__ == "__main__":
    #app.debug = True
    #port = int(os.environ.get("PORT", 8070))
    #app.run(host="0.0.0.0", port=port)
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)
