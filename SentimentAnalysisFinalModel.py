import pandas as pd
path = r'D:\ML and DL Coading practce\Projects with Deployments\NLP Sentiment Analysis with Flask\sentiment.tsv'

data = pd.read_csv(path, sep='\t')
data.columns = ['label', 'text']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r'@\w+|#\w+', ' ', text)
    text = re.sub(r'http?:\S+',' ', text)
    text = re.sub(r'[^A-Za-z\s\']', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

data['clean_text'] = data['text'].apply(clean_text)

# Split text by whitespace to tokenize
data['tokenized_text'] = data['clean_text'].apply(lambda x: ' '.join(x.split()))

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
data['removed_stopwords'] = data['tokenized_text'].apply(lambda x: ' '.join([w for w in x.split() if w not in stop_words]))

from nltk.stem import WordNetLemmatizer
data['after_lemmatization'] = data['removed_stopwords'].apply(lambda x: ' '.join([WordNetLemmatizer().lemmatize(w) for w in x.split()]))

from sklearn.feature_extraction.text import TfidfVectorizer
tfidfvector = TfidfVectorizer()
text_tfidfvector = tfidfvector.fit_transform(data['after_lemmatization'])

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(text_tfidfvector, data['label'], test_size=0.2, random_state=42)
lr.fit(X_train_tfidf, y_train_tfidf)

model_name = 'lr_model.pkl'
tfidf_name = 'tfidf_vectorizer.pkl'
import pickle
pickle.dump(lr, open(model_name, 'wb'))
pickle.dump(tfidfvector, open(tfidf_name, 'wb'))