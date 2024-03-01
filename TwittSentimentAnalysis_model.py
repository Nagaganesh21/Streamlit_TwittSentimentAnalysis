import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')

# Load the TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", 'rb') as f:
    tfidf = pickle.load(f)

# Load the model
with open("lr_model.pkl", 'rb') as f:
    model = pickle.load(f)

# Define preprocess_message function
def preprocess_message(message):
    # Stopwords removal
    stop_words = set(stopwords.words('english'))
    message = ' '.join([word for word in message.split() if word.lower() not in stop_words])
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    message = ' '.join([lemmatizer.lemmatize(word) for word in message.split()])
    return message

# Define main function
def main():
    st.title("Tweet Sentiment Analysis")

    # Get user input
    message = st.text_input("Enter your tweet:")

    if st.button("Predict"):
        # Preprocess the input message
        preprocessed_message = preprocess_message(message)
        data = [preprocessed_message]

        # Transform the preprocessed message using TF-IDF vectorizer
        vect = tfidf.transform(data)

        # Predict sentiment
        prediction = model.predict(vect)

        # Display prediction result
        if prediction == 1:
            st.write("Prediction is: Positive Tweet")
        else:
            st.write("Prediction is: Negative Tweet")

if __name__ == "__main__":
    main()
