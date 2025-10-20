import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "SMSSpamCollection")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "model_results.csv")

def load_data():
    print("📂 Loading full SMS Spam Collection dataset...")
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "text"])
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df

def compare_models():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label_num"], test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    results = []

    for name, clf in models.items():
        print(f"\n🔹 Training {name}...")
        model = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", clf)
        ])
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        print(f"{name} Accuracy: {acc:.3f}")
        print(classification_report(y_test, preds))

        results.append({
            "Model": name,
            "Accuracy": round(acc, 3),
            "Precision": round(prec, 3),
            "Recall": round(rec, 3),
            "F1-Score": round(f1, 3)
        })

    # Save table to CSV
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\n📊 Model comparison table saved to: {RESULTS_PATH}\n")
    print(results_df)

    # Save best model
    best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
    print(f"\n🏆 Best Model: {best_model_name} with Accuracy: {results_df['Accuracy'].max():.3f}")

    best_model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", models[best_model_name])
    ])
    best_model.fit(df["text"], df["label_num"])
    model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(model_dir, "spam_model.joblib"))
    print(f"✅ Best model saved as spam_model.joblib")

if __name__ == "__main__":
    print("🚀 Comparing multiple models...")
    compare_models()
