import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

def run_cv_classical(X_train, y_train, vectorizers, models, scoring="f1"):
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for vec_name, vectorizer in vectorizers.items():
        for model_name, model in models.items():
            pipe = Pipeline([("vec", vectorizer), ("clf", model)])

            if model_name == "LogReg":
                param_grid = {"vec__max_features": [3000, 10000], "clf__C": [0.1, 1, 10]}
            elif model_name == "RandomForest":
                param_grid = {"vec__max_features": [3000, None], "clf__n_estimators": [200, 500]}
            else:
                param_grid = {}

            gs = GridSearchCV(pipe, param_grid, scoring=scoring, cv=cv, n_jobs=-1, verbose=1)
            gs.fit(X_train, y_train)

            results.append({
                "Feature": vec_name,
                "Model": model_name,
                "Best Params": gs.best_params_,
                "CV Score": gs.best_score_
            })
    return pd.DataFrame(results)

def run_cv_bert(X_train, y_train, get_bert_embeddings, models, scoring="f1"):
    X_train_bert = get_bert_embeddings(X_train.tolist())
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
        gs = GridSearchCV(pipe, param_grid={"clf__C": [0.1, 1, 10]} if model_name=="LogReg" else {},
                          scoring=scoring, cv=cv, n_jobs=-1, verbose=1)
        gs.fit(X_train_bert, y_train)

        results.append({
            "Feature": "BERT",
            "Model": model_name,
            "Best Params": gs.best_params_,
            "CV Score": gs.best_score_
        })
    return pd.DataFrame(results)
