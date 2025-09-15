#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# Define stopwords
STOPWORDS = set(stopwords.words('english'))

# Paths to resources
MODEL_PATH = r"C:\Users\Abidingvoice\Python\Project\Sentiment Analysis\models\sentiment_model.pkl"
VECTORIZER_PATH = r"C:\Users\Abidingvoice\Python\Project\Sentiment Analysis\models\vectorizer.pkl"
DATA_PATH = r"C:\Users\Abidingvoice\Python\Project\Sentiment Analysis\Data\customer_reviews.csv"

# Load the Saved model and vectorizer
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

with open(VECTORIZER_PATH, 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Load the dataset
df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1')

# Preprocess function
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

# Initialize the Tkinter app
root = tk.Tk()
root.title("Sentiment Analysis Feedback")

# Load emojis
happy_emoji = ImageTk.PhotoImage(Image.open(r"C:\Users\Abidingvoice\Python\Project\Sentiment Analysis\images\happy_emoji.png").resize((100, 100)))
neutral_emoji = ImageTk.PhotoImage(Image.open(r"C:\Users\Abidingvoice\Python\Project\Sentiment Analysis\images\neutral_emoji.png").resize((100, 100)))
angry_emoji = ImageTk.PhotoImage(Image.open(r"C:\Users\Abidingvoice\Python\Project\Sentiment Analysis\images\angry_emoji.png").resize((100, 100)))

# Create widgets
emoji_label = tk.Label(root)
emoji_label.grid(row=0, column=0, padx=10, pady=10)

review_label = tk.Label(root, wraplength=300, justify="center")
review_label.grid(row=1, column=0, padx=10, pady=10)

# Create a button to move to the next review
current_index = [0]

# Function for updating feedback
def update_feedback(index):
    review_text = df['Detailed Review'].iloc[index]
    processed_review = preprocess_review(review_text)
    X_review = vectorizer.transform([processed_review]).toarray()
    predicted_sentiment = model.predict(X_review)[0]
    
    # Map sentiment to emoji
    emoji_map = {1: happy_emoji, 0: neutral_emoji, -1: angry_emoji}
    emoji_image = emoji_map.get(predicted_sentiment, neutral_emoji)
    
    # Update emoji and review text
    emoji_label.config(image=emoji_image)
    emoji_label.image = emoji_image
    review_label.config(text=review_text)

# Function for navigating reviews
def next_review():
    current_index[0] = (current_index[0] + 1) % len(df)
    update_feedback(current_index[0])

# Next review button
next_button = ttk.Button(root, text="Next Review", command=next_review)
next_button.grid(row=2, column=0, padx=10, pady=10)

# Initialize with the first review
update_feedback(current_index[0])

# Start the Tkinter main loop
root.mainloop()


# In[ ]:




