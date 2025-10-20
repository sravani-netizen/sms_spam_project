import os
import joblib

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "..", "model", "spam_model.joblib")
model = joblib.load(model_path)

def predict_message(message):
    prediction = model.predict([message])[0]
    return "🚨 SPAM" if prediction == 1 else "✅ HAM (Not Spam)"

if __name__ == "__main__":
    while True:
        msg = input("\nEnter a message to check (or 'exit' to quit): ")
        if msg.lower() == "exit":
            break
        print(predict_message(msg))
