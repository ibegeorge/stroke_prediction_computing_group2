# 🧠 Stroke Prediction Machine Learning Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

## 📖 Table of Contents
- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Methodology & Pipeline](#methodology--pipeline)
- [Models Implemented](#models-implemented)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [Contributors](#contributors)
- [Acknowledgments](#acknowledgments)

---

## 🎯 About the Project
This repository contains a comprehensive machine learning pipeline developed to predict the likelihood of a stroke based on various health and demographic indicators. Built as part of the **GW4 Computing Group 2** project, this work explores data preprocessing, exploratory data analysis (EDA), and the comparative performance of multiple classification algorithms to identify the most reliable model for stroke prediction.

A detailed project report, including visualizations (confusion matrices, ROC curves) and in-depth analysis, is available in [`GW4_COMPUTING_GROUP2.pdf`](GW4_COMPUTING_GROUP2.pdf).

---

## 📊 Dataset
The project utilizes the `health_stroke_dataset.csv`, a well-known healthcare dataset containing demographic and health metrics of patients. 

**Key Features Include:**
- **Demographics:** `gender`, `age`, `ever_married`, `Residence_type`
- **Medical History:** `hypertension`, `heart_disease`
- **Lifestyle:** `work_type`, `smoking_status`
- **Health Metrics:** `avg_glucose_level`, `bmi`
- **Target Variable:** `stroke` (Binary: 1 if the patient had a stroke, 0 otherwise)

---

## ⚙️ Methodology & Pipeline
The machine learning pipeline follows a structured approach to ensure data quality and model reliability:

1. **Data Cleaning:** 
   - Identification and removal of duplicate records.
   - Missing value imputation: `bmi` missing values are filled with the median to mitigate the impact of outliers.
2. **Outlier Detection:** Applied the Interquartile Range (IQR) method on numerical columns (`age`, `avg_glucose_level`, `bmi`) to identify and handle anomalies.
3. **Feature Transformation:** 
   - Log transformation (`np.log1p`) applied to `avg_glucose_level` to handle right-skewed data.
4. **Feature Encoding:**
   - **Label Encoding:** Applied to binary categorical variables (`gender`, `ever_married`, `Residence_type`).
   - **One-Hot Encoding:** Applied to nominal categorical variables (`work_type`, `smoking_status`) using `pd.get_dummies` with `drop_first=True` to avoid multicollinearity.

---

## 🤖 Models Implemented
To find the most robust predictor, the project implements and compares four distinct classification algorithms, including hyperparameter tuning via `GridSearchCV` where applicable:

1. **Logistic Regression** (Optimized with GridSearchCV)
2. **Mixed Naive Bayes** (Specifically chosen to handle the mix of continuous and categorical predictors)
3. **Support Vector Machine (SVM)** (Both baseline and hyperparameter-optimized versions)
4. **K-Nearest Neighbors (KNN)** (Both baseline and hyperparameter-optimized versions)

---

## 📈 Evaluation Metrics
Models are rigorously evaluated using a comprehensive suite of classification metrics to ensure a balanced assessment, especially important for potentially imbalanced medical datasets:
- **Accuracy**
- **Precision**
- **Recall (Sensitivity)**
- **F1-Score**
- **ROC-AUC Score**
- **Confusion Matrices** (Visualized for each model)

A final comparison table is generated in the script to rank the models based on these metrics.

---

## 📁 Project Structure
```text
stroke_prediction_computing_group2/
│
├── health_stroke_dataset.csv       # Raw dataset used for training and evaluation
├── stroke_pred.py                  # Main Python script containing the ML pipeline
├── GW4_COMPUTING_GROUP2.pdf        # Comprehensive project report with graphs and analysis
└── README.md                       # Project documentation (this file)
```

---

## 🚀 Installation & Usage

### Prerequisites
Ensure you have Python 3.8+ installed. You will need the following libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib` / `seaborn` (for visualizations, if extended)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/ibegeorge/stroke_prediction_computing_group2.git
   cd stroke_prediction_computing_group2
   ```
2. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Run the main script:
   ```bash
   python stroke_pred.py
   ```

---

## 👥 Contributors
This project was developed collaboratively by:
- **ibegeorge** ([@ibegeorge](https://github.com/ibegeorge))
- **judeesther2000** ([@judeesther2000](https://github.com/judeesther2000))

---

## 📄 License
This project is intended for academic and educational purposes. Please refer to the original dataset source for any specific data usage licenses.

---

## 🙏 Acknowledgments
- The dataset is sourced from publicly available healthcare repositories (commonly associated with Kaggle's Stroke Prediction Dataset).
- Special thanks to the GW4 Computing module instructors and peers for feedback and guidance.

---

### 💡 Next Steps / Future Improvements
- Implement advanced handling for class imbalance (e.g., SMOTE, class weights).
- Explore ensemble methods like Random Forest or XGBoost for potential performance gains.
- Deploy the best-performing model as a simple web application using Streamlit or Flask.
