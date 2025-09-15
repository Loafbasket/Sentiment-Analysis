#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer



# Load the saved model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
     model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
     vectorizer = pickle.load(vec_file)

# Load and preprocess the dataset
reviews = pd.read_csv(r"C:\Users\Abidingvoice\Python\Project\Sentiment Analysis\Data\customer_reviews.csv", encoding='ISO-8859-1')

# Define stopwords
STOPWORDS = set(stopwords.words('english'))

# Preprocess function from sentiment_analysis.py
def preprocess_review(text):
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

# Preprocess the first review and predict sentiment
first_review = reviews['Detailed Review'].iloc[0]
processed_review = preprocess_review(first_review)
X_review = vectorizer.transform([processed_review]).toarray()
predicted_sentiment = model.predict(X_review)[0]

# Load emojis
happy_emoji = mpimg.imread('happy_emoji.png')
neutral_emoji = mpimg.imread('neutral_emoji.png')
angry_emoji = mpimg.imread('angry_emoji.png')



# Map sentiment to emoji
emoji_map = {1: happy_emoji, 0: neutral_emoji, -1: angry_emoji} # Assuming label 1 = positive, 0 = neutral, -1 = negative
emoji = emoji_map.get(predicted_sentiment, neutral_emoji)

# Create a figure to display the emoji and the review text
fig, ax = plt.subplots(figsize=(6, 4))

# Hide the axes
ax.axis('off')

# Display the emoji
ax.imshow(emoji, extent=[0.2, 0.8, 0.4, 1])

# Display the review text below the emoji
ax.text(0.5, 0.2, first_review, fontsize=12, ha='center', va='center', wrap=True, bbox=dict(facecolor='lightgray'))

# Show the plot
plt.tight_layout()
plt.show()


# In[ ]:




