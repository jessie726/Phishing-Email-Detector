import pandas as pd
import re
from sklearn.utils import shuffle
from tqdm import tqdm
tqdm.pandas()  
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'\s+', ' ', text) 
    text = ' '.join(text.split()) 
    text = text.lower() 
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stemmed = stemmer.stem(text)
    text = lemmatizer.lemmatize(stemmed)
    return text
