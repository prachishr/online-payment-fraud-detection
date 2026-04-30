# 💳 Online Payment Fraud Detection System


## 📌 Project Overview

This project focuses on detecting fraudulent financial transactions using machine learning. The system analyzes transaction details and predicts whether a transaction is fraudulent or legitimate.

A complete end-to-end pipeline is implemented, including data preprocessing, model training, evaluation, and deployment using a Streamlit web application.

---

## 🚀 Features

* Detects fraudulent transactions in real-time
* Uses machine learning (XGBoost) for prediction
* Handles imbalanced data effectively
* Interactive web app using Streamlit
* Displays fraud probability along with prediction

---

## 🧠 Machine Learning Approach

The project follows a structured ML pipeline:

1. Data Cleaning & Preprocessing
2. Feature Engineering
3. Handling Class Imbalance (SMOTE & scale_pos_weight)
4. Model Training:

   * Logistic Regression
   * Random Forest
   * XGBoost (Final Model)
5. Threshold Tuning for optimal performance
6. Model Evaluation using Precision, Recall, and F1-score

---

## 📊 Model Performance

Final model: **XGBoost**

* Recall: ~0.68
* Precision: ~0.24
* F1 Score: ~0.35

The model is optimized to maximize recall, ensuring that more fraudulent transactions are detected.

---

## 🔍 Key Insights

* Fraud detection depends on behavioral patterns, not just transaction amount
* Balance inconsistencies play a major role in identifying fraud
* Threshold tuning helps balance precision and recall
* High recall is preferred to avoid missing fraud cases

---

## 🛠️ Tech Stack

* Python
* Pandas & NumPy
* Scikit-learn
* XGBoost
* Streamlit

---

## 📂 Project Structure

```
fraud-detection/
│
├── app.py
├── fraud_model.pkl
├── encoder.pkl
├── features.pkl
├── requirements.txt
├── README.md
├── model_training.ipynb
```

---

## ▶️ How to Run Locally

1. Clone the repository:

```
git clone https://github.com/your-username/fraud-detection.git
```

2. Navigate to the project folder:

```
cd fraud-detection
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the Streamlit app:

```
streamlit run app.py
```

---

## 🌐 Deployment

The application can be deployed using Streamlit Community Cloud for free hosting.

---

## 📸 Demo

(Add your app screenshot here after deployment)

---

## 📌 Conclusion

This project demonstrates how machine learning can be used to detect fraudulent financial transactions by learning complex patterns in transaction behavior.

---

## 👨‍💻 Author

Prachi Sharma

