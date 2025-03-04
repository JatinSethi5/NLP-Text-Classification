import pandas as pd
import re
import string
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tokenizers import Tokenizer
import pickle




def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    # call stop words and remove them
    stop_words = stopwords.words('english') 
    removed_stopwords_text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    # Perform stemming
    stemmer = nltk.SnowballStemmer("english")
    return ' '.join(stemmer.stem(word) for word in removed_stopwords_text.split(' '))