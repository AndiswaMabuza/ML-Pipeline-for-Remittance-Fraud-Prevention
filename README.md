# ML Pipeline for Remittance Fraud Prevention

## Created By

Andiswa Mabuza| Amabuza53@gmail.com

## Project Overview

This repository presents an end-to-end Machine Learning (ML) pipeline designed to detect fraudulent transactions within a remittance system. Financial institutions and remittance companies face the critical challenge of preventing fraud without overburdening their operations with false alarms. This project tackles this challenge head-on by developing an optimized fraud detection solution that achieves **exceptional performance in both fraud recall and precision**.

The core objective was to build a robust system that could **catch virtually all fraudulent transactions (high recall)** while simultaneously **minimizing the flagging of legitimate transactions as fraudulent (high precision)**. This delicate balance is crucial for protecting customers and the institution from financial loss, and for ensuring smooth, efficient operational workflows.

## The Challenge: Navigating Extreme Imbalance

Fraud detection datasets are notoriously imbalanced, with fraudulent transactions typically representing less than 1% of the total. This poses a significant challenge for traditional machine learning models, which often default to classifying most instances as the majority class (non-fraud), leading to misleadingly high accuracy but catastrophically low fraud detection rates.

My initial experiments quickly highlighted this issue:

### Initial Approach: Logistic Regression Baseline

I began with a Logistic Regression model, employing SMOTE (Synthetic Minority Over-sampling Technique) to address the class imbalance in the training data.

**Initial Performance (Logistic Regression with Default Threshold):**

  * **Precision (Fraud):** 0.18
  * **Recall (Fraud):** 1.00
  * **F1-Score (Fraud):** 0.30
  * **ROC AUC Score:** 0.9993

**Analysis:** While the model achieved perfect recall (catching all fraud), its precision was unacceptably low. A precision of 0.18 means that for every 10 transactions flagged as fraud, 8 were actually legitimate. This would lead to an overwhelming number of false positives, rendering the system operationally inefficient and costly.

### Strategic Optimization: Threshold Tuning for Balance

Recognizing the excellent ROC AUC score, I understood that the model was highly capable of *ranking* transactions by their fraud probability, even if its default binary classification was flawed. Our next step was to optimize the classification threshold, moving away from the default 0.5. I achieved this by analyzing the Precision-Recall curve and selecting the threshold that maximized the F1-score, a metric that provides a harmonic mean of precision and recall.

**Improved Performance (Logistic Regression with Optimized Threshold):**

  * **Precision (Fraud):** 0.45
  * **Recall (Fraud):** 0.90
  * **F1-Score (Fraud):** 0.60
  * **ROC AUC Score:** 0.9993

**Analysis:** This optimization significantly improved precision while maintaining a high recall. The system became far more practical, reducing false alarms and achieving a much better balance between fraud detection and operational efficiency.

### The Breakthrough: Leveraging CatBoost for Superior Performance

To achieve truly exceptional results and push the boundaries of performance, I transitioned to **CatBoost**, a state-of-the-art gradient boosting library renowned for its robust performance on tabular data and, crucially, its native handling of categorical features. This eliminated the need for explicit one-hot encoding, simplifying the pipeline and improving model interpretability and accuracy.

The data preprocessing was refined using a `ColumnTransformer` to scale numerical features, while categorical features were passed directly to CatBoost. `SMOTENC` (SMOTE Nominal Continuous) was employed for handling imbalance in the mixed-type dataset. The F1-score optimization strategy was reapplied to the CatBoost model's probabilities.

## Final Model Performance (CatBoost Pipeline)

The final CatBoost-powered ML pipeline delivers outstanding performance, meeting and exceeding the project's ambitious goals:

  * **Fraud Recall (Class 1):** **1.00**
      * **Impact:** Absolutely no fraudulent transactions are missed. This provides the highest level of security against financial losses.
  * **Fraud Precision (Class 1):** **0.77**
      * **Impact:** When the system flags a transaction, it is correct 77% of the time, drastically reducing false positives and optimizing the workload for fraud review teams.
  * **Fraud F1-Score (Class 1):** **0.87**
      * **Impact:** This exceptionally high F1-score reflects a near-perfect balance between catching all fraud and minimizing false alarms, a critical achievement in fraud detection.
  * **ROC AUC Score:** **0.9998**
      * **Impact:** The model exhibits an almost flawless ability to distinguish between legitimate and fraudulent transactions.

## Key Features & Technical Stack

  * **Model:** CatBoost Classifier
  * **Imbalance Handling:** SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous features)
  * **Preprocessing:** `sklearn.preprocessing.StandardScaler` via `sklearn.compose.ColumnTransformer` for numerical features; native categorical handling by CatBoost.
  * **Optimization:** F1-score based optimal classification threshold tuning.
  * **Persistency:** Model and preprocessing artifacts saved using `joblib` for seamless deployment.
  * **Interactive Demo:** A Streamlit web application (`app.py`) for live risk assessment.

**Libraries Used:**

  * `pandas`
  * `numpy`
  * `scikit-learn`
  * `catboost`
  * `imblearn`
  * `joblib`
  * `streamlit`
  * `plotly`

## Project Structure

```
.
├── app.py                      # Streamlit web application for interactive demo
├── fraud_detection_model.pkl   # Trained CatBoost model
├── data_preprocessor.pkl       # ColumnTransformer for preprocessing
├── classification_threshold.pkl# Optimized classification threshold
├── all_feature_names.pkl       # List of all feature names in correct order
├── numerical_cols.pkl          # List of numerical feature names
├── categorical_cols.pkl        # List of categorical feature names
└── requirements.txt            # Python dependencies
```

## How to Run the Demo

You can view a live demo of this application here: **[remittance-fraud-prevention.streamlit.app](https://remittance-fraud-prevention.streamlit.app)**

```

## Conclusion

This project demonstrates a robust and highly effective ML pipeline for remittance fraud prevention. By focusing on critical business metrics, strategically optimizing model performance, and leveraging advanced machine learning techniques like CatBoost, we've developed a solution that is both incredibly accurate and operationally viable. This pipeline stands as a testament to the power of data-driven decision-making in safeguarding financial operations.
