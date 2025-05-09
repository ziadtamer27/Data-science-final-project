# Stroke Risk Prediction - Healthcare Predictive Analytics Project

This project is part of the IBM AI & Data Science Track , focused on developing a predictive model to assess stroke risk based on various health parameters. The system provides healthcare professionals with data-driven insights for patient risk prediction and informed decision-making.

## Project Overview

The Healthcare Predictive Analytics project uses machine learning to forecast stroke risk, helping healthcare providers identify patients who may benefit from preventive interventions. The project follows a structured approach through multiple milestones:

1. **Data Collection, Exploration, and Preprocessing**
2. **Data Analysis and Visualization**
3. **Predictive Model Development and Optimization**
4. **MLOps, Deployment, and Monitoring**
5. **Final Documentation and Presentation**

## Repository Structure

```
├── templates/
│   ├── form.html                  # Input form for patient data
│   ├── result.html                # Results display page
│   └── error.html                 # Error handling page
├── healthcare_dataset.csv         # Original dataset
├── clean_data_1.csv               # Cleaned dataset (Milestone 1 output)
├── clean_data_2.csv               # Enhanced dataset (Milestone 2 output)
├── Milestone_1.ipynb              # Data Collection, Exploration & Preprocessing
├── Milestone_2.ipynb              # Data Analysis, Visualization & Feature Engineering 
├── Milestone_3.ipynb              # Model Development & Optimization
├── Milestone_4.ipynb              # MLOps, Deployment & Monitoring 
├── main.py                        # FastAPI application entry point
├── xgboost_model.pkl              # Trained and optimized model
├── requirements.txt               # Project dependencies
└── README.md                      # This file
```

## Data Description

The dataset contains various health parameters that can influence stroke risk:

- **age**: Patient's age
- **gender**: Patient's gender (Male/Female)
- **hypertension**: Whether the patient has hypertension (0/1)
- **heart_disease**: Whether the patient has heart disease (0/1)
- **ever_married**: Marriage status (Yes/No)
- **work_type**: Type of employment (Private, Self-employed, Govt_job, Children, Never_worked)
- **Residence_type**: Living environment (Urban/Rural)
- **avg_glucose_level**: Average glucose level in blood
- **bmi**: Body Mass Index
- **smoking_status**: Smoking habits (never smoked, formerly smoked, smokes)
- **stroke**: Target variable - whether the patient had a stroke (1) or not (0)

## Key Features

- **Exploratory Data Analysis**: Comprehensive analysis of health metrics and their relationships
- **Feature Engineering**: Creation of categorical features from continuous variables
- **Model Selection**: Comparison of multiple ML algorithms (XGBoost selected as final model)
- **Web Application**: FastAPI-based interface for healthcare professionals to input patient data
- **Risk Assessment**: Immediate prediction of stroke risk with probability percentage

## Installation and Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/stroke-risk-prediction.git
cd stroke-risk-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the FastAPI application:
```bash
uvicorn main:app --reload
```

5. Open your browser and navigate to http://127.0.0.1:8000

## Usage

1. Access the web interface at http://127.0.0.1:8000
2. Enter patient health parameters in the form
3. Submit the form to get stroke risk prediction
4. View the prediction result with risk percentage

## Requirements

See the `requirements.txt` file for the complete list of dependencies. Main libraries include:

- FastAPI
- Numpy
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Joblib
- Uvicorn
- Jinja2

## Project Milestones

### Milestone 1: Data Collection, Exploration, and Preprocessing
- Obtained healthcare dataset with relevant features
- Conducted initial EDA to understand data distribution
- Cleaned data by addressing missing values and outliers

### Milestone 2: Data Analysis and Visualization
- Performed in-depth correlation analysis between health metrics
- Created visualizations including heatmaps and distribution plots
- Engineered new features to improve model performance

### Milestone 3: Predictive Model Development and Optimization
- Evaluated multiple algorithms (Logistic Regression, Random Forest, XGBoost)
- Used techniques like SMOTE to address class imbalance
- Performed hyperparameter tuning to optimize model performance

### Milestone 4: MLOps, Deployment, and Monitoring
- Deployed model as a web application using FastAPI
- Created user-friendly interface for healthcare professionals
- Implemented error handling and validation

## Model Performance
After trying different models:
| Model | ROC AUC | Precision_0 | Recall_0 | F1_0 | Precision_1 | Recall_1 | F1_1 |
|-------|---------|-------------|----------|------|-------------|----------|------|
| XGBoost | 0.978548 | 0.97 | 0.89 | 0.93 | 0.86 | 0.96 | 0.91 |
| Gradient Boosting | 0.964925 | 0.93 | 0.91 | 0.92 | 0.87 | 0.90 | 0.89 |
| Random Forest | 0.912017 | 0.94 | 0.73 | 0.82 | 0.71 | 0.93 | 0.81 |
| Logistic | 0.876605 | 0.85 | 0.78 | 0.81 | 0.72 | 0.81 | 0.76 |

The XGBoost model was selected based on its superior performance across multiple metrics:
Classification Report For XGBoost:
              Classification Report For XGBoost:
              precision    recall  f1-score    support
           0       0.97      0.89      0.93        972
           1       0.86      0.96      0.91        681
    accuracy                           0.92       1653
   macro avg       0.91      0.92      0.92       1653
weighted avg       0.92      0.92      0.92       1653

AUC-ROC For XGBoost: 0.978

*Note: These metrics represent the performance on the test dataset*

## Future Improvements

- Implement continuous model monitoring for performance drift
- Add more advanced feature engineering techniques
- Develop a more comprehensive dashboard for healthcare analytics
- Integrate with electronic health record systems

## Contributors

- Mazen Mohamed Hemdan
- Ziad Tamer Ibrahim
- Ebrahim Benbella ElSayed
- Joumana Mohamed
- Basmalla Hussien
