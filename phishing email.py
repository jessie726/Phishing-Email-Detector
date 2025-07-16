import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import re
files = ["CEAS_08.csv","Enron.csv","Ling.csv","Nazario.csv","Nigerian_Fraud.csv","SpamAssasin.csv"]
dfs = [pd.read_csv(fname) for fname in files]
all_data = pd.concat(dfs, ignore_index = True)
print(all_data.info())
num_spam = (all_data.label==1).sum()
num_ham = (all_data.label==0).sum()
print(f"The number of spam emails is {num_spam}, and the number of ham emails is {num_ham}") 
all_data['text_combined'] = all_data['subject'].fillna('') + ' ' + all_data['body'].fillna('')
all_data['text_length'] = all_data['text_combined'].apply(len)


def contains_url(text):
    return int(bool(re.search(r'http[s]?://|www\.', text)))

all_data['has_url'] = all_data['text_combined'].apply(contains_url)

final_cols = ['text_combined', 'text_length', 'has_url', 'label']
all_data = all_data[final_cols]
print(all_data.info())
ham  = all_data[all_data.label == 0]
spam = all_data[all_data.label == 1]
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
# Data Cleaning
from tqdm import tqdm
tqdm.pandas()  
import nltk
from nltk.corpus import stopwords

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
    return text

def preprocess(df, column_name):
    df['text_cleaned'] = df[column_name].progress_apply(clean_text)
    return df
preprocess(train_set, 'text_combined')
preprocess(test_set, 'text_combined')
from sklearn.model_selection import StratifiedKFold

X = train_set['text_cleaned']
y = train_set['label']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
mean_fpr = np.linspace(0, 1, 100)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model        import LogisticRegression
from sklearn.ensemble            import RandomForestClassifier
from sklearn.naive_bayes         import MultinomialNB
from sklearn.metrics             import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection     import StratifiedKFold

# set up your data and splitter
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# define the feature transforms you want to try
vectorizers = {
    "TF-IDF": TfidfVectorizer(max_features=1000, ngram_range=(1,2)),
    "BoW":    CountVectorizer(max_features=1000, ngram_range=(1,2)),
}

# define the models you want to try
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Multinomial NB":      MultinomialNB(),
}

for vec_name, vectorizer in vectorizers.items():
    print(f"\n=== {vec_name} Cross-Validation ===")
    for model_name, model in models.items():
        print(f"\n-- Model: {model_name} --")
        fold_metrics = {"acc":[], "prec":[], "rec":[], "f1":[]}

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_val = X.iloc[train_idx],   X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx],   y.iloc[val_idx]

            # fit vectorizer & transform
            X_train_vec = vectorizer.fit_transform(X_train)
            X_val_vec   = vectorizer.transform(X_val)

            # train & predict
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_val_vec)

            # record metrics
            fold_metrics["acc"].append(accuracy_score(y_val, y_pred))
            fold_metrics["prec"].append(precision_score(y_val, y_pred, zero_division=0))
            fold_metrics["rec"].append(recall_score(y_val, y_pred, zero_division=0))
            fold_metrics["f1"].append(f1_score(y_val, y_pred, zero_division=0))

            print(f"Fold {fold:>2} • Acc: {fold_metrics['acc'][-1]:.3f}  "
                  f"P: {fold_metrics['prec'][-1]:.3f}  "
                  f"R: {fold_metrics['rec'][-1]:.3f}  "
                  f"F1: {fold_metrics['f1'][-1]:.3f}")

        # summary across folds
        print(f"→ Mean • Acc: {np.mean(fold_metrics['acc']):.3f}  "
              f"P: {np.mean(fold_metrics['prec']):.3f}  "
              f"R: {np.mean(fold_metrics['rec']):.3f}  "
              f"F1: {np.mean(fold_metrics['f1']):.3f}")
    
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score
)
import matplotlib.pyplot as plt

# === TF-IDF Pipeline ===
pipeline_tfidf = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1,2))),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline_tfidf.fit(train_set['text_cleaned'], train_set['label'])
y_pred_tfidf = pipeline_tfidf.predict(test_set['text_cleaned'])
y_prob_tfidf = pipeline_tfidf.predict_proba(test_set['text_cleaned'])[:, 1]

print("=== TF-IDF Classification Report ===")
print(classification_report(test_set['label'], y_pred_tfidf))
print("TF-IDF Test Accuracy:", accuracy_score(test_set['label'], y_pred_tfidf))

# Confusion Matrix - TF-IDF
cm_tfidf = confusion_matrix(test_set['label'], y_pred_tfidf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_tfidf, display_labels=["Ham", "Spam"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix: TF-IDF")
plt.show()

# === BoW Pipeline ===
pipeline_bow = Pipeline([
    ('bow', CountVectorizer(max_features=1000, ngram_range=(1,2))),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline_bow.fit(train_set['text_cleaned'], train_set['label'])
y_pred_bow = pipeline_bow.predict(test_set['text_cleaned'])
y_prob_bow = pipeline_bow.predict_proba(test_set['text_cleaned'])[:, 1]

print("=== BoW Classification Report ===")
print(classification_report(test_set['label'], y_pred_bow))
print("BoW Test Accuracy:", accuracy_score(test_set['label'], y_pred_bow))

# Confusion Matrix - BoW
cm_bow = confusion_matrix(test_set['label'], y_pred_bow)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_bow, display_labels=["Ham", "Spam"])
disp.plot(cmap="Greens")
plt.title("Confusion Matrix: Bag of Words")
plt.show()

# === ROC Curves ===
fpr_tfidf, tpr_tfidf, _ = roc_curve(test_set['label'], y_prob_tfidf)
auc_tfidf = roc_auc_score(test_set['label'], y_prob_tfidf)

fpr_bow, tpr_bow, _ = roc_curve(test_set['label'], y_prob_bow)
auc_bow = roc_auc_score(test_set['label'], y_prob_bow)

plt.figure(figsize=(8,6))
plt.plot(fpr_tfidf, tpr_tfidf, label=f'TF-IDF AUC = {auc_tfidf:.3f}', color='blue')
plt.plot(fpr_bow, tpr_bow, label=f'BoW AUC = {auc_bow:.3f}', color='green')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison: TF-IDF vs BoW')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

import joblib
joblib.dump(pipeline_bow, "spam_detector.pkl")
