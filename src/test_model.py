import os
import joblib
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "spam_model.joblib")
RESULTS_PATH = os.path.join(BASE_DIR, "..", "data", "results.csv")

def load_model():
    print("📦 Loading saved model...")
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!\n")
    return model

def test_samples(model):
    # Predefined test messages
    test_messages = [
        "Congratulations! You've won a free iPhone!",
        "Hey, are we meeting for lunch today?",
        "Claim your prize now! Limited time offer!",
        "Reminder: your electricity bill is due tomorrow.",
        "Win cash now!!! Just click the link below.",
        "Let's catch up at the café later.",
        "URGENT! You have been selected for a reward.",
        "Call me when you’re free.",
        "Get free access to premium content now!",
        "Your OTP is 564321. Do not share it with anyone."
    ]

    print("🧪 Running test samples...\n")
    results = []

    for msg in test_messages:
        prediction = model.predict([msg])[0]
        label = "Spam" if prediction == 1 else "Ham"
        print(f"{msg}  ➡️  {label}")
        results.append({"Message": msg, "Prediction": label})

    # Save results to CSV
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\n✅ Test results saved to: {RESULTS_PATH}\n")

def interactive_mode(model):
    print("💬 Enter your own messages to test (type 'exit' to quit):\n")
    while True:
        message = input("Enter a message: ")
        if message.lower() == "exit":
            print("👋 Exiting...")
            break
        prediction = model.predict([message])[0]
        label = "Spam" if prediction == 1 else "Ham"
        print(f"➡️ Prediction: {label}\n")

if __name__ == "__main__":
    model = load_model()
    test_samples(model)
    interactive_mode(model)
