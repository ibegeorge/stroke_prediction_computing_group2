import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,classification_report,roc_curve)

df_original = pd.read_csv("health_stroke_dataset.csv")
df = df_original.copy()

print(df.head())
print(df.shape)
print(df.columns)

#checking for missing values
print("\nchecking for missing values ... ")
for column in df.columns:
    missing_values = df[column].isnull().sum()
    if missing_values > 0:
        print(f"Found {missing_values} missing values in {column}")
#checking for duplicate records
print("\nchecking for duplicate records ... ")
duplicates = df.duplicated().sum()
print(f"Found {duplicates} duplicate records")


