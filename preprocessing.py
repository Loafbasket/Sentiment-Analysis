#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

# Ensure necessary resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Define stopwords
STOPWORDS = set(stopwords.words('english'))

def preprocess_review(text):
    """
    Clean and preprocess the review text.
    """
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize and lemmatize
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word, 'v') for word in words if word not in STOPWORDS]
    return " ".join(words)


# In[ ]:




