# 📌 Healthcare Predictive Analytics - Stroke Prediction - Milsetone 1

## 📖 Overview
This project focuses on predicting the likelihood of a stroke based on various health metrics using **Machine Learning**. The dataset includes crucial health indicators such as age, BMI, glucose levels, and smoking status. By leveraging data preprocessing, feature engineering, and model training, we aim to build a robust predictive model to assist healthcare professionals in early stroke detection.

## 📂 Dataset Description
The dataset consists of patient information with various health attributes:
- **Age**: Numerical variable (categorized into Young, Middle-aged, Elderly)
- **BMI (Body Mass Index)**: Continuous variable (categorized into Underweight, Normal, Overweight, Obese)
- **Average Glucose Level**: Continuous variable (categorized into Low, Normal, Prediabetes, Diabetes, Severe Diabetes)
- **Smoking Status**: Categorical (Never Smoked, Formerly Smoked, Smokes, Unknown)
- **Other Health Factors**: Hypertension, Heart Disease, etc.
- **Target Variable**: Stroke (1: Stroke occurred, 0: No stroke)

## ⚙️ Data Preprocessing Steps
### 1️⃣ **Handling Missing Values**
- Imputed missing BMI and glucose values using **median imputation**.
- Replaced `Unknown` values in **smoking status** proportionally based on the existing distribution.

### 2️⃣ **Outlier Detection & Treatment**
- Used **IQR (Interquartile Range)** to remove outliers in BMI (capped at 45) and glucose levels (adjusted for diabetic patients).
- Ensured that unrealistic BMI values (e.g., 97) were handled properly.

### 3️⃣ **Feature Engineering**
- **Age Categories**: Converted numerical ages into three groups (Young, Middle-aged, Elderly).
- **BMI Categories**: Classified into Underweight, Normal, Overweight, and Obese.
- **Glucose Level Categories**: Divided into Low, Normal, Prediabetes, Diabetes, Severe Diabetes.

### 4️⃣ **Encoding Categorical Variables**
- Used **Label Encoding** for ordered categories (Age, BMI, Glucose Levels).
- Applied **One-Hot Encoding** for non-ordinal variables (Smoking Status, etc.).

## 📊 Data Visualization & Insights
To understand the dataset better, we implemented various visualizations:
- **Histogram of Glucose Levels** to analyze the distribution across different categories.
- **Bar Plots for Age & BMI Categories** to compare distributions.
- **Pie Charts & Count Plots for Smoking Status** to observe imbalances in data.

## 🚀 Next Steps
- Model Development & Training
- Further **feature selection** to optimize model performance.
- Improve handling of **class imbalance** (e.g., using SMOTE or weighted models).
- Deploy the trained model as an API or interactive dashboard for real-world application.

---
### 🎯 **Conclusion**
This project provides a data-driven approach to **stroke prediction** using patient health metrics. By applying feature engineering and model optimization techniques, we aim to improve healthcare diagnostics and support medical decision-making.

---
📌 **Contributors:** Mazen Mohamed, Ziad Tamer, Ibrahim Benbella, Jomana Mohamed, Basmala Hussein  
📌 **Project Type:** Healthcare Predictive Analytics  
📌 **Technologies Used:** Python, Pandas, Scikit-Learn, Matplotlib, Seaborn, XGBoost  
📌 **Dataset Source:** Collected & Preprocessed Healthcare Data from kaggle  
       🔗 https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset



