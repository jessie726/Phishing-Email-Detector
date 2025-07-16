from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load("spam_detector.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        subject = request.form.get("subject", "")
        body = request.form.get("body", "")
        email_text = subject + " " + body
        pred = model.predict([email_text])[0]
        prediction = "Spam" if pred == 1 else "Ham"
    return render_template("index.html", prediction=prediction)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
