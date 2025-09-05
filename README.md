# Phishing-Email-Detector
I trained multiple models (LSTM, BERT / TF-IDF / BOW + Logistic Regression, etc.).

The LSTM model gave the best performance.

Problem⚠️: Render free tier is only 512 MB RAM, and my LSTM/BERT models are too heavy → leads to Worker Timeout (SIGKILL, Out of Memory) errors.

Thus, I'm deploying the TF-IDF + Logistic Regression pipeline (final_tfidf.pkl) because:

It’s lightweight.

Can be loaded quickly with joblib or pickle.

Runs well on Render.

This is mainly for getting a working visible website.
