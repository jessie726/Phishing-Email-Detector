import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB


# =====================
# Classical Models (TF-IDF, BoW)
# =====================
def run_cv_classical(X_train, y_train, vectorizers, models, scoring="f1"):
    """
    Run cross-validation for classical vectorizers (TF-IDF, BoW) + models.
    """
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for vec_name, vectorizer in vectorizers.items():
        for model_name, model in models.items():
            print(f"Running CV for {vec_name} + {model_name}...")

            pipe = Pipeline([("vec", vectorizer), ("clf", model)])

            # Small param grids to avoid huge memory usage
            if model_name == "LogReg":
                param_grid = {"vec__max_features": [3000, 10000],
                              "clf__C": [0.01, 0.1, 1, 10]}
            elif model_name == "RandomForest":
                param_grid = {"vec__max_features": [3000],
                              "clf__n_estimators": [200, 500]}
            else:
                param_grid = {}

            gs = GridSearchCV(pipe,
                              param_grid=param_grid,
                              scoring=scoring,
                              cv=cv,
                              n_jobs=-1,
                              verbose=1)
            gs.fit(X_train, y_train)

            results.append({
                "Feature": vec_name,
                "Model": model_name,
                "Best Params": gs.best_params_,
                "CV Score": gs.best_score_
            })

    return pd.DataFrame(results)


# =====================
# BERT Embeddings
# =====================
def run_cv_bert(X_train, y_train, get_bert_embeddings, models, scoring="f1",
                batch_size=32, max_len=128, device=None):
    """
    Run cross-validation using BERT embeddings (batched + GPU support).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Generating BERT embeddings on {device}...")
    X_train_bert = get_bert_embeddings(X_train.tolist(),
                                       batch_size=batch_size,
                                       max_len=max_len,
                                       device=device)

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model in models.items():
        print(f"Running CV for BERT + {model_name}...")

        if model_name == "NaiveBayes":
            print("MultinomialNB not supported for BERT embeddings, using GaussianNB instead.")
            model = GaussianNB()

        pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])

        # Param grid only for Logistic Regression
        param_grid = {"clf__C": [0.1, 1, 10]} if model_name == "LogReg" else {}

        gs = GridSearchCV(pipe,
                          param_grid=param_grid,
                          scoring=scoring,
                          cv=cv,
                          n_jobs=1,    # keep memory usage safe
                          verbose=1)
        gs.fit(X_train_bert, y_train)

        results.append({
            "Feature": "BERT",
            "Model": model_name,
            "Best Params": gs.best_params_,
            "CV Score": gs.best_score_
        })

    return pd.DataFrame(results)
