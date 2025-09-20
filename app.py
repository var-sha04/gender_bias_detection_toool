from flask import Flask, render_template, request
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

nltk.download('stopwords')

app = Flask(__name__)

# Load dataset
df = pd.read_csv('vat_baner1.csv')

# Text cleaning function
def clean_text(Sentence):
    Sentence = str(Sentence).lower()
    Sentence = re.sub(r"[^a-zA-Z\s]", "", Sentence)
    tokens = Sentence.split()
    stop_words = set(stopwords.words("english"))
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered) if filtered else "emptytext"

# Apply cleaning
df['clean_text'] = df['Sentence'].apply(clean_text)
df = df[df['clean_text'].str.strip() != ""]  # Remove empty ones

# Features and labels
X = df['clean_text']
y = df['Label']  # make sure it's lowercase 'label' as in your CSV

# Vectorize the input
vectorizer = TfidfVectorizer(max_features=1000)
X_vec = vectorizer.fit_transform(X)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_vec, y)

# Save model and vectorizer (optional if you want to reuse later)
joblib.dump(model, 'gender_bias_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Prediction function
def predict_bias(sentence):
    cleaned = clean_text(sentence)
    if cleaned == "emptytext":
        return "⚠️ Invalid input. Please enter meaningful text."
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return "⚠️ Gender Bias Detected" if prediction == 1 else "✅ No Gender Bias Detected"

# Web interface
@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    user_text = ""
    if request.method == "POST":
        user_text = request.form["user_text"]
        result = predict_bias(user_text)
    return render_template("index.html", result=result, user_text=user_text)

if __name__ == "__main__":
    app.run(debug=True)
