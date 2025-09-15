#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import pickle

# Downloading NLTK stopwords
nltk.download('stopwords')

# Define stopwords
STOPWORDS = set(stopwords.words('english'))

# Paths to resources
MODEL_PATH = r"C:\Users\Abidingvoice\Python\Project\Sentiment Analysis\models\sentiment_model.pkl"
VECTORIZER_PATH = r"C:\Users\Abidingvoice\Python\Project\Sentiment Analysis\models\vectorizer.pkl"
DATA_PATH = r"C:\Users\Abidingvoice\Python\Project\Sentiment Analysis\Data\customer_reviews.csv"

# Load the dataset with the correct encoding
df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1')

# Dropping duplicates in the dataset based on available columns

df = df.drop_duplicates(subset={"Sl.No", "Headline", "Detailed Review"}, keep='first')

# Function to clean the text data (as per your sentiment analysis process)

def doTextCleaning(review):
    review = re.sub('<.*?>', '', review)
    review = re.sub(r'[^\w\s]', '', review)
    review = review.lower()
    review = review.split()
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(word, 'v') for word in review if word not in STOPWORDS]
    return " ".join(review)


# Preprocess the reviews
corpus = []
for index, row in tqdm(df.iterrows()):
    review = doTextCleaning(row['Detailed Review'])
    corpus.append(review)

# Convert text into numerical format using CountVectorizer (Bag of Words)
cv = CountVectorizer(ngram_range=(1,3), max_features=5000)
x = cv.fit_transform(corpus).toarray()
# y = [1 if i % 2 == 0 else for i in range(len(x))]
y = [1 if i % 2 == 0 else 0 for i in range(len(x))]

# Split dataset into training and testing sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

# Train a Naïve Bayes model
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Save the trained model and vectorizer
with open('sentiment_model.pkl', 'wb') as model_file:
     pickle.dump(classifier, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
     pickle.dump(cv, vec_file)

# Predict the test set results and print evaluation
y_pred = classifier.predict(x_test)
print(r"Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(r"Classification Report:\n", classification_report(y_test, y_pred))


# In[ ]:




