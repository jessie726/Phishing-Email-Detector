import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import re
from etl.extract import extract
from etl.transform import clean_text
from tqdm import tqdm
tqdm.pandas()
files = [
        "data/CEAS_08.csv",
        "data/Enron.csv",
        "data/Ling.csv",
        "data/Nazario.csv",
        "data/Nigerian_Fraud.csv",
        "data/SpamAssasin.csv"
    ]
all_data = extract(files)
print(all_data)
print(all_data.info())

ham  = all_data[all_data.label == 0]
spam = all_data[all_data.label == 1]
num_spam = (all_data.label==1).sum()
num_ham = (all_data.label==0).sum()
# Data Splitting
spam = shuffle(spam, random_state=42)
ham  = shuffle(ham, random_state=42)

n_test_spam = (num_spam + num_ham) // 10 // 10 
n_test_ham = n_test_spam * 9

test_spam = spam.iloc[:n_test_spam]
test_ham  = ham.iloc[:n_test_ham]
test_set = pd.concat([test_spam, test_ham], ignore_index=True).reset_index(drop=True)
test_set = shuffle(test_set, random_state=42).reset_index(drop=True)

train_spam = spam.iloc[n_test_spam:]
train_ham  = ham.iloc[n_test_ham:]
train_set = pd.concat([train_spam, train_ham], ignore_index=True).reset_index(drop=True)
train_set = shuffle(train_set, random_state=42).reset_index(drop=True)
print(f"Train set: {train_set.shape[0]} samples (spam: {sum(train_set.label==1)}, ham: {sum(train_set.label==0)})")
print(f"Test set:  {test_set.shape[0]} samples (spam: {sum(test_set.label==1)}, ham: {sum(test_set.label==0)})")


train_set['text_cleaned'] = train_set['text_combined'].progress_apply(clean_text)
test_set['text_cleaned'] = test_set['text_combined'].progress_apply(clean_text)
print(train_set)
print(test_set)

from features.vectorizers import tfidf, bow, get_bert_embeddings
from modeling.classical import get_models
from evaluation.cv import run_cv_classical, run_cv_bert
import pandas as pd


def main(X_train, y_train):
    vectorizers = {"TF-IDF": tfidf(), "BoW": bow()}

    df_classical = run_cv_classical(X_train, y_train, vectorizers, get_models())
    df_bert = run_cv_bert(X_train, y_train, get_bert_embeddings, get_models())

    results = pd.concat([df_classical, df_bert], ignore_index=True)
    print(results.sort_values("CV Score", ascending=False))
    return results

X_train = train_set['text_cleaned']
y_train = train_set['label']
results = main(X_train, y_train)
print(results)