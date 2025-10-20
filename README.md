# 📱 SMS Spam Detection using Machine Learning

A machine learning project that detects **spam or ham (non-spam)** messages using multiple models and a **Streamlit web app** for real-time testing.

---

## 🚀 Overview

This project explores various machine learning models to classify SMS messages as **spam** or **ham**.  
The system is trained on the classic **SMS Spam Collection Dataset** and compares different models to find the best one.

The final **Streamlit app** allows users to enter a message and instantly see whether it’s spam or not.

---

## 🧠 Models Compared

| Model | Accuracy | Remarks |
|:------|:----------:|:--------|
| Logistic Regression | 96.3% | Fast and interpretable |
| Naive Bayes | 97.2% | Performs well on text data |
| SVM (Support Vector Machine) | 🏆 **98.3%** | Best performer overall |
| Decision Tree | 96.4% | Slightly overfits |
| Random Forest | 98.0% | Strong ensemble model |

✅ **Best Model Saved:** `spam_model.joblib`  
✅ **Used for prediction in the Streamlit app**

---

## 🧩 Folder Structure

