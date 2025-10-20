import streamlit as st
import joblib
import os

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "model", "spam_model.joblib")
model = joblib.load(model_path)

st.set_page_config(page_title="📱 SMS Spam Detector", page_icon="✉️")

st.title("📱 SMS Spam Detector")
st.markdown("Detect whether an SMS message is **Spam** or **Not Spam (Ham)** using a trained ML model.")

# Dropdown for sample messages
st.subheader("🔍 Try with Example Messages")
sample_messages = {
    "Hey, are you coming to college tomorrow?": "Ham (Legit)",
    "Please call me when you reach home.": "Ham (Legit)",
    "Congratulations! You’ve won a free iPhone! Click here to claim your prize.": "Spam",
    "You have been selected for a cash reward. Reply WIN to claim now!": "Spam",
    "Your Amazon order has been shipped successfully.": "Ham (Legit)",
    "URGENT! Your account is suspended. Verify immediately at fakebank.com.": "Spam",
    "Free entry in 2 a weekly contest! Text WIN to 80085 now!": "Spam",
    "Can we have lunch together today?": "Ham (Legit)",
    "Claim your Rs. 10,000 reward card today — limited time offer!": "Spam"
}

selected_message = st.selectbox("Select a test message:", list(sample_messages.keys()))
st.write(f"💬 *Example Type:* **{sample_messages[selected_message]}**")

st.markdown("---")
st.subheader("✉️ Or Enter Your Own Message")
user_input = st.text_area("Enter your SMS message here:", height=100)

# Choose which message to test
final_message = user_input.strip() if user_input.strip() else selected_message

if st.button("🚀 Predict"):
    prediction = model.predict([final_message])[0]
    if prediction == 1:
        st.error("🚫 This message is **SPAM**!")
    else:
        st.success("✅ This message is **NOT SPAM (Ham)**.")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit and Scikit-learn")
