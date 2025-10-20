print("🚀 Training script started...")
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Path to dataset
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "SMSSpamCollection")

def load_data(path=DATA_PATH):
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df

def train_model():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label_num"], test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "spam_model.joblib")
    joblib.dump(model, model_path)
    print(f"✅ Model saved at: {model_path}")

if __name__ == "__main__":
    train_model()
