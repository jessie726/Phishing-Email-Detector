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

'''
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

# Save CV results to CSV
results.to_csv("cv_results.csv", index=False)

# Save CV results as pickle (keeps Python objects, e.g. dicts in "Best Params")
results.to_pickle("cv_results.pkl")

print("✅ Cross-validation results saved as cv_results.csv and cv_results.pkl")
'''
X_train = train_set['text_cleaned']
y_train = train_set['label']

X_test = test_set['text_cleaned']
y_test = test_set['label']

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

import pandas as pd, ast

# Load CV results
cv_results = pd.read_csv("cv_results.csv")

# Convert string dicts into Python dicts
cv_results["Best Params"] = cv_results["Best Params"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith("{") else {}
)

# Pick the best model per feature type
best_per_feature = cv_results.loc[cv_results.groupby("Feature")["CV Score"].idxmax()]
print(best_per_feature)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

'''final_models = {}

for _, row in best_per_feature.iterrows():
    feature, model_name, params = row["Feature"], row["Model"], row["Best Params"]

    if feature == "TF-IDF" and model_name == "LogReg":
        model = Pipeline([
            ("vec", TfidfVectorizer(max_features=params.get("vec__max_features", 10000),
                                    ngram_range=(1,2))),
            ("clf", LogisticRegression(C=params.get("clf__C", 1),
                                       max_iter=5000, random_state=42))
        ])
        model.fit(X_train, y_train)
        joblib.dump(model, "final_tfidf.pkl")
        final_models["TF-IDF"] = model

    elif feature == "BoW" and model_name == "LogReg":
        model = Pipeline([
            ("vec", CountVectorizer(max_features=params.get("vec__max_features", 5000),
                                    ngram_range=(1,2))),
            ("clf", RandomForestClassifier(
                n_estimators=params.get("clf__n_estimators", 100),
                random_state=42
            ))
        ])
        model.fit(X_train, y_train)
        joblib.dump(model, "final_bow.pkl")
        final_models["BoW"] = model

    elif feature == "BERT":
        # Build embeddings for train & test
        X_train_bert = get_bert_embeddings(list(X_train))
        X_test_bert = get_bert_embeddings(list(X_test))

        if model_name == "SVM":
            model = SVC(probability=True, C=params.get("clf__C", 1))
        else:
            model = LogisticRegression(C=params.get("clf__C", 1), max_iter=5000)

        model.fit(X_train_bert, y_train)
        joblib.dump(model, "final_bert.pkl")
        final_models["BERT"] = (model, X_test_bert)  # store test embeddings
'''
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Tokenizer
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X_train)

'''# Save tokenizer
with open("final_models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Tokenizer recreated and saved as tokenizer.pkl")
'''

# Convert to padded sequences
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=120)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=120)

'''from modeling.deep import build_lstm, build_gru, build_cnn, build_bilstm

deep_models = {
    "LSTM": build_lstm(vocab_size=20000, embedding_dim=100, maxlen=120, units=128, dropout=0.2),
    "GRU": build_gru(vocab_size=20000, embedding_dim=100, maxlen=120, units=128, dropout=0.2),
    "CNN": build_cnn(vocab_size=20000, embedding_dim=100, maxlen=120),
    "BiLSTM": build_bilstm(vocab_size=20000, embedding_dim=100, maxlen=120, units=128, dropout=0.2)
}

for name, model in deep_models.items():
    print(f"Training {name}...")
    model.fit(
        X_train_seq, y_train,
        epochs=5, batch_size=64,
        validation_split=0.1,
        verbose=1
    )
    model.save(f"final_{name.lower()}.h5")'''

from evaluation.test import evaluate_model
import joblib

# Reload models
tfidf_model = joblib.load("final_models/final_tfidf.pkl")
bow_model = joblib.load("final_models/final_bow.pkl")
bert_model = joblib.load("final_models/final_bert.pkl")

# Classical models
evaluate_model(tfidf_model, X_test, y_test, "final_models/TF-IDF + LogReg")
evaluate_model(bow_model, X_test, y_test, "final_models/BoW + RandomForest")

# BERT (needs embeddings)
X_test_bert = get_bert_embeddings(list(X_test))
evaluate_model(bert_model, X_test_bert, y_test, "BERT + SVM")

from tensorflow.keras.models import load_model

# Reload deep models
lstm_model = load_model("final_models/final_lstm.h5")
gru_model = load_model("final_models/final_gru.h5")
cnn_model = load_model("final_models/final_cnn.h5")
bilstm_model = load_model("final_models/final_bilstm.h5")

# Evaluate on test sequences
evaluate_model(lstm_model, X_test_seq, y_test, "LSTM", is_keras=True)
evaluate_model(gru_model, X_test_seq, y_test, "GRU", is_keras=True)
evaluate_model(cnn_model, X_test_seq, y_test, "CNN", is_keras=True)
evaluate_model(bilstm_model, X_test_seq, y_test, "BiLSTM", is_keras=True)
