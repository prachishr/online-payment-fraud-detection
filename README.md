# 💳 Online Payment Fraud Detection System

## 🚀 Live Demo  
👉 [Click here to run the app](https://online-payment-fraud-detection-4qetcnhc5z5hkj7n3shzgx.streamlit.app/)


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

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 99.92% |
| Precision | 75.4%  |
| Recall    | 55.1%  |
| F1 Score  | 63.7%  |

### Cross Validation

* 5-Fold Stratified Cross Validation Mean F1 Score: **53.8%**

### Model Optimization

The final model performance was improved through:

* Feature Engineering
* Class Imbalance Handling using `scale_pos_weight`
* Threshold Optimization
* Removal of harmful outlier clipping on financial transaction features

The optimal classification threshold was found to be **0.9**, improving the F1-score from approximately **0.42** to **0.64**.


## 🔍 Key Insights

* Fraud detection depends on transaction behavior patterns rather than transaction amount alone
* Balance inconsistencies and account balance changes are strong fraud indicators
* Financial transaction outliers contain useful fraud signals and should not always be removed
* Threshold optimization significantly reduced false positives while improving overall F1-score
* Precision, Recall, and F1-score are more reliable evaluation metrics than Accuracy for highly imbalanced fraud datasets

---

## 🛠️ Tech Stack

* Python
* Pandas & NumPy
* Seaborn & Matplotlib
* Scikit-learn
* XGBoost
* Streamlit

---


## 📂 Project Structure

```text
online-payment-fraud-detection/
│
├── app.py
├── xgb_fraud_model.pkl
├── encoder.pkl
├── features.pkl
├── requirements.txt
├── README.md
├── Online_Payment_Fraud_Detection.ipynb
```


## ▶️ How to Run Locally

1. Clone the repository:

```
git clone https://github.com/prachishr/online-payment-fraud-detection.git
```

2. Navigate to the project folder:

```
cd online-payment-fraud-detection
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the Streamlit app:


[Run the Streamlit App](https://online-payment-fraud-detection-4qetcnhc5z5hkj7n3shzgx.streamlit.app/)



---

## 🌐 Deployment

The application can be deployed using Streamlit Community Cloud for free hosting.

---
## 📸 Demo

<img width="1911" height="965" alt="image" src="https://github.com/user-attachments/assets/1ec47886-41af-4ae4-8a8d-ffcde1a9f782" />


---

## 📌 Conclusion

This project demonstrates how machine learning can be used to detect fraudulent financial transactions by learning complex patterns in transaction behavior.

---

## 👨‍💻 Author

Prachi Sharma

