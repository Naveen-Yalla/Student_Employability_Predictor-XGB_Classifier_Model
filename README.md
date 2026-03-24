# Student Employability Prediction

## 📊 Project Overview
A complete ML pipeline predicting student employability from soft skills ratings. Achieves 91% F1-score using XGBoost with Streamlit deployment on Hugging Face.
Predicts "Employable" vs "Less Employable" students using 7 ordinal soft skills features (2-5 scale) from 2982 student records. End-to-end pipeline: EDA → Model Building → Deployment.

## Tech Stack:
Python, scikit-learn, XGBoost, Streamlit, Pandas, joblib, Hugging Face

## 🗃️ Dataset
- Source: Student-Employability.csv (2982 records)
- Features: 7 soft skills (GENERAL APPEARANCE, MENTAL_ALERTNESS, etc.) + STUDENT PERFORMANCE RATING
- Target: Binary CLASS (Employable=1, LessEmployable=0)
- Split: 75/25 train-test (2236/746)

## 🛠️ Pipeline
1. EDA → Distributions, class imbalance visualization
2. Preprocessing → StandardScaler Pipeline
3. Model Tuning → GridSearchCV (5-fold CV, F1 scoring)  
4. Evaluation → Classification metrics + Feature importance
5. Deployment → Streamlit app → Hugging Face

## 🎯 Key Results

| Model | Test F1-Score | Test Accuracy | Precision (Class 1) | Recall (Class 1) | Confusion Matrix |
|-------|---------------|---------------|-------------------|-----------------|------------------|
| **XGBoost** | **0.91** | **0.90** | **0.90** | **0.92** | <pre>[[297  40]<br>[ 34 375]]</pre> |
| Random Forest | 0.90 | 0.88 | 0.84 | 0.97 | <pre>[[260  77]<br>[ 12 397]]</pre> |

#### **Top Feature: MENTAL_ALERTNESS (16.8% importance)**

## 📈 Model Performance Highlights

**XGBoost Pipeline:**
- F1-Score: 0.91 (test set)
- Precision: 0.90, Recall: 0.92 (Class 1)
- Best params: n_estimators=200, max_depth=10
- Deployment-ready with StandardScaler preprocessing

## 🚀 Live Demo 👇
click here:
[[Student Employability Predictor]](https://huggingface.co/spaces/Naveen-Yalla/Student_Employability)

## 🔄 How to Run Locally
1. Clone repository
2. Install dependencies → pip install -r requirements.txt
3. Run Streamlit app → streamlit run app.py
