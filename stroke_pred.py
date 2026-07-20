import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression 
from mixed_naive_bayes import MixedNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,classification_report,roc_curve)

df_original = pd.read_csv("health_stroke_dataset.csv")
df = df_original.copy()
df1 = df.copy()
print(df.head())
print(df.shape)
print(df.columns)

#checking for missing values
print("\nchecking for missing values ... ")
for column in df.columns:
    missing_values = df[column].isnull().sum()
    if missing_values > 0:
        print(f"Found {missing_values} missing values in {column}")
print("\nReplacing missing bmi values with the median because of presence of outliers ...")
df['bmi'] = df["bmi"].fillna(df["bmi"].median())
#checking for duplicate records
print("\nchecking for duplicate records ... ")
duplicates = df.duplicated().sum()
print(f"Found {duplicates} duplicate records")

#checking for outliers
print("checking for outliers in the dataset ...")

numerical_columns = ["age","avg_glucose_level","bmi"]

for col in numerical_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    outliers = df[(df[col] < lower) | (df[col]> upper)]
    print(f"{col}: {len(outliers)}")
print("visualising the outliers ...")

plt.figure(figsize=(12,6))
df[numerical_columns].boxplot()
plt.xticks(rotation=45)
plt.savefig("outliers.jpeg")
plt.show()


print("\nchecking for if the numerical cols are highly skewed...")
numerical_columns = ["age", "avg_glucose_level", "bmi"]
skewness_results = []
for col in numerical_columns:
    skew_value = df[col].skew()

    skewness_results.append({
        "Variable": col,
        "Skewness": skew_value
    })

skewness_df = pd.DataFrame(skewness_results)

print(skewness_df)

print("log transforming avg_glucose_level ... ")
df["avg_glucose_level"] = np.log1p(df["avg_glucose_level"])

print("Encoding categorical columns ... ")
binary_columns = ["gender","ever_married","Residence_type"]

for column in binary_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])

df = pd.get_dummies(
    df, columns=["work_type", "smoking_status"],
    drop_first=True, dtype=float
)

# Calculate the correlation matrix
correlation_matrix = df.drop(columns=["id"]).corr()

# Print the matrix
print("\nCorrelation matrix:")
print(correlation_matrix.round(2).head())

stroke_correlation = (correlation_matrix["stroke"].drop("stroke").sort_values(ascending=False))

print("\nCorrelation of each variable with stroke:")
print(stroke_correlation)

stroke_correlation.sort_values().plot(
    kind="barh",
    figsize=(10, 8)
)

plt.title("Correlation of Predictors with Stroke")
plt.xlabel("Correlation coefficient")
plt.ylabel("Predictor")
plt.axvline(x=0)
plt.tight_layout()
plt.savefig("corr_predictors_vs_stroke.jpeg")
plt.show()

# Extracting features and target
X = df.drop(columns=["id","stroke"])
y = df["stroke"]

#splitting dataset into train and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, shuffle=True, random_state=42, stratify=y)

#scaling X_train and X_test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training a Logistic Regression Model
logistic_model = LogisticRegression(random_state=42, max_iter=1000)
param_grid = [
    # L1 and L2 using liblinear
    {
        "solver": ["liblinear"],
        "penalty": ["l1", "l2"],
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "class_weight": [None, "balanced"]
    },
    # L2 using lbfgs
    {
        "solver": ["lbfgs"],
        "penalty": ["l2"],
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "class_weight": [None, "balanced"]
    },
    # L2 using newton-cg
    {
        "solver": ["newton-cg"],
        "penalty": ["l2"],
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "class_weight": [None, "balanced"]
    },
    # L2 using sag
    {
        "solver": ["sag"],
        "penalty": ["l2"],
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "class_weight": [None, "balanced"]
    },
    # L1, L2 and Elastic Net using saga
    {
        "solver": ["saga"],
        "penalty": ["l1", "l2", "elasticnet"],
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "l1_ratio": [0.1, 0.5, 0.9],
        "class_weight": [None, "balanced"]
    }

]
# Grid search
grid_search = GridSearchCV(
    estimator=logistic_model,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=2
)
grid_search.fit(
    X_train_scaled,
    y_train
)
print("Best Parameters:")
print(grid_search.best_params_)
print("\nBest Cross Validation Score:")
print(grid_search.best_score_)

best_logistic_model = grid_search.best_estimator_


logistic_pred = best_logistic_model.predict(X_test_scaled)
logistic_probability = best_logistic_model.predict_proba(X_test_scaled)[:,1]
print(logistic_probability[:10].round(4))
print("\nAccuracy:")
print(accuracy_score(y_test,logistic_pred))
print("\nPrecision:")
print(precision_score(y_test,logistic_pred))
print("\nRecall:")
print(recall_score(y_test,logistic_pred))
print("\nF1 Score:")
print(f1_score(y_test,logistic_pred))
print("\nROC-AUC:")
print(roc_auc_score(y_test,logistic_probability))
print("\nClassification Report:")
print(classification_report(y_test,logistic_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test,logistic_pred))

# applying decision threshold
threshold = 0.5
y_threshold = (logistic_probability >= threshold).astype(int)
print(y_threshold)
accuracy_score(y_test, y_threshold)
precision_score(y_test, y_threshold)
recall_score(y_test, y_threshold)
f1_score(y_test, y_threshold)
print(classification_report(y_test, y_threshold))

# Confusion matrix
cm = confusion_matrix(y_test, y_threshold)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Logistics Confusion Metrics")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("logistics_model_confusion_matrix.jpeg")
plt.show()
#ROC-AUC 
auc = roc_auc_score(y_test, logistic_probability)
print(f"ROC-AUC: {auc}")

#ROC curve
fpr,tpr,thresholds = roc_curve(y_test, logistic_probability)
plt.plot(fpr, tpr)
plt.title("Logistic Regression ROC Curve")
plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("logistic_regression_roc_curve.jpeg")
plt.show()

# Threshold analysis
threshold_results = []
thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
for threshold in thresholds:
    # convert probability to class prediction
    y_threshold_pred = (logistic_probability >= threshold).astype(int)
    threshold_results.append({"Threshold": threshold, "Accuracy": accuracy_score(y_test, y_threshold_pred),
        "Precision": precision_score(y_test, y_threshold_pred, zero_division=0),
        "Recall": recall_score(y_test, y_threshold_pred),
        "F1-score": f1_score(y_test, y_threshold_pred, zero_division=0)})
threshold_df = pd.DataFrame(threshold_results)
print(threshold_df)

f1_scores = threshold_df["F1-score"]
best_threshold = thresholds[np.argmax(f1_scores)]
best_f1 = max(f1_scores)
print("Best Threshold:", best_threshold)
print("Best F1-score:", best_f1)

y_best_pred = (logistic_probability >= best_threshold).astype(int)
cm1 = confusion_matrix(y_test, y_best_pred)
print(classification_report(y_test, y_best_pred))

# Confusion matrix for the best threshold
sns.heatmap(cm1, annot=True, fmt='d')
plt.title("Logistics Optimal Confusion Metrics")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("optimal_logistics_model_confusion_matrix.jpeg")
plt.show()

X_sm = sm.add_constant(X_train)
logit = sm.Logit(y_train, X_sm)
result = logit.fit(method = "lbfgs", maxiter=1000)
print(result.summary())
params = result.params
conf = result.conf_int()
interpretation = pd.DataFrame({"Coefficient": params, "Odds Ratio": np.exp(params), "Lower CI": np.exp(conf[0]), "Upper CI": np.exp(conf[1]), "p-value": result.pvalues})


# Function to generate conclusions
def interpret_logistic(row):
    OR = row["Odds Ratio"]
    p = row["p-value"]
    lower = row["Lower CI"]
    upper = row["Upper CI"]
    # Calculate percentage change
    if OR > 1:
        percentage = (OR - 1) * 100
        effect = f"increased odds by approximately {percentage:.4f}%"
    elif OR < 1:
        percentage = (1 - OR) * 100
        effect = f"reduced odds by approximately {percentage:.4f}%"
    else:
        effect = "no change in odds"

    # Check significance
    if (p < 0.05) and (lower > 1 or upper < 1):
        if OR > 1:
            conclusion = (
                f"Significant association: this variable was associated "
                f"with {effect} of stroke "
                f"(OR={OR:.4f}, 95% CI={lower:.4f}-{upper:.4f}, "
                f"p={p:.4f})."
            )
        else:
            conclusion = (
                f"Significant association: this variable was associated "
                f"with {effect} of stroke "
                f"(OR={OR:.4f}, 95% CI={lower:.4f}-{upper:.4f}, "
                f"p={p:.4f})."
            )
    else:
        conclusion = (
            f"No statistically significant association was found "
            f"(OR={OR:.4f}, 95% CI={lower:.4f}-{upper:.4f}, "
            f"p={p:.4f})."
        )
    return conclusion
# Apply interpretation
interpretation["Conclusion"] = interpretation.apply(interpret_logistic,axis=1)

interpretation["Percentage Change (%)"] = np.where(
    interpretation["Odds Ratio"] > 1,
    (interpretation["Odds Ratio"] - 1) * 100,
    (1 - interpretation["Odds Ratio"]) * 100
)
# Display results
print(interpretation.drop("const"))

# NAIVE BAYES
#The dataset contains heterogeneous predictors consisting of continuous clinical measurements and categorical patient characteristics
#So, we will use Mixed Naive Bayes 
df1["bmi"] = df1["bmi"].fillna(df1["bmi"].median())
categorical_columns = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

for col in categorical_columns:
    encoder = LabelEncoder()
    df1[col] = encoder.fit_transform(df1[col])
X_mnb = df1.drop(columns=["id", "stroke"])
y_mnb = df1["stroke"]    
print(X_mnb.columns)
# columns with categorical features
categorical_features = [0,2,3,4,5,6,9]
# splitting X and y for mixed naive bayes
X_mnb_train,X_mnb_test,y_mnb_train,y_mnb_test = train_test_split(X_mnb,y_mnb,test_size=0.2,random_state=42,stratify=y_mnb)

mnb_model = MixedNB(
    categorical_features=categorical_features
)
mnb_model.fit(
    X_mnb_train,
    y_mnb_train
)
# Evaluation
mnb_prediction = mnb_model.predict(X_mnb_test)
mnb_probability = mnb_model.predict_proba(X_mnb_test)[:,1]
print(mnb_probability)
print("Naive Bayes Accuracy:", accuracy_score(y_mnb_test,mnb_prediction))
print("Naive Bayes Precision:", precision_score(y_mnb_test,mnb_prediction))
print("Naive Bayes Recall:", recall_score(y_mnb_test,mnb_prediction))
print("Naive Bayes F1:", f1_score(y_mnb_test,mnb_prediction))
print("Naive Bayes ROC-AUC:", roc_auc_score(y_mnb_test,mnb_probability))
print(classification_report(y_mnb_test,mnb_prediction))
mnb_cm = confusion_matrix(y_mnb_test, mnb_prediction)
sns.heatmap(mnb_cm, annot=True, fmt='d')
plt.title("Mixed Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("mnb_confusion_matrix.jpeg")
plt.show()

# Hyperprameter tunning for Mixed Naive Bayes

alphas = [0.01,0.1,0.5,1,5,10]
results=[]
for alpha in alphas:
    model = MixedNB(
        categorical_features=categorical_features,
        alpha=alpha
    )
    model.fit(
        X_mnb_train,
        y_mnb_train
    )
    probability = model.predict_proba(
        X_mnb_test
    )[:,1]
    auc = roc_auc_score(
        y_mnb_test,
        probability
    )
    results.append(
        {
            "alpha":alpha,
            "ROC_AUC":auc
        }
    )
print(results)
print("\nFrom the result, the best alpha = 5 because it has the highest ROC_AUC = 0.8133539...")
mnb_model1 = MixedNB(
    categorical_features=categorical_features,
    alpha=5
)

mnb_model1.fit(
    X_mnb_train,
    y_mnb_train
)
# Evaluation after tunning
optimal_mnb_prediction = mnb_model1.predict(X_mnb_test)
optimal_mnb_probability = mnb_model1.predict_proba(X_mnb_test)[:,1]
print(optimal_mnb_probability)
print("Naive Bayes Accuracy:", accuracy_score(y_mnb_test,optimal_mnb_prediction))
print("Naive Bayes Precision:", precision_score(y_mnb_test,optimal_mnb_prediction))
print("Naive Bayes Recall:", recall_score(y_mnb_test,optimal_mnb_prediction))
print("Naive Bayes F1:", f1_score(y_mnb_test,optimal_mnb_prediction))
print("Naive Bayes ROC-AUC:", roc_auc_score(y_mnb_test,optimal_mnb_probability))
print(classification_report(y_mnb_test,optimal_mnb_prediction))
optimal_mnb_cm = confusion_matrix(y_mnb_test, optimal_mnb_prediction)
sns.heatmap(optimal_mnb_cm, annot=True, fmt='d')
plt.title("Optimal Mixed Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("optimal_mnb_confusion_matrix.jpeg")
plt.show()

fpr, tpr, thresholds = roc_curve(
    y_mnb_test,
    optimal_mnb_probability
)
plt.figure(figsize=(8,6))
plt.plot(
    fpr,
    tpr,
    label=f"MixedNB (AUC = {auc:.3f})"
)
plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Mixed Naive Bayes")
plt.legend()
plt.savefig("roc_curve_mnb.jpeg")
plt.show()

# Threshold analysis
mnb_thresholds = [0.1,0.2,0.3,0.4,0.5]
for threshold in mnb_thresholds:
    mnb_predictiont = (
        optimal_mnb_probability >= threshold
    ).astype(int)
    print("Threshold:", threshold)
    print(
        "Recall:",
        recall_score(y_mnb_test,mnb_predictiont)
    )
    print(
        "F1:",
        f1_score(y_mnb_test,mnb_predictiont)
    )
    
print(classification_report(y_mnb_test,mnb_predictiont))    
cm2 = confusion_matrix(y_test, mnb_predictiont)
plt.title("Final Mixed Naive Bayes Confusion Matrix")
sns.heatmap(cm2, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("final_mnb_confusion_matrix.jpeg")
plt.show()

# Non-Probabilistic Classification Models 
# 1. Support Vector Machine (SVM) 
svm_model = SVC(
    kernel = "rbf",
    probability=True,
    class_weight="balanced",
    random_state=42
)
svm_model.fit(X_train_scaled,y_train)
svm_prediction = svm_model.predict(X_test_scaled)
svm_probability = svm_model.predict_proba(X_test_scaled)[:,1]
print("SVM Accuracy:",accuracy_score(y_test,svm_prediction))
print("SVM Precision:",precision_score(y_test,svm_prediction))
print("SVM Recall:",recall_score(y_test,svm_prediction))
print("SVM F1:",f1_score(y_test,svm_prediction))
print("SVM ROC-AUC:",roc_auc_score(y_test,svm_probability))
print(classification_report(y_test,svm_prediction))
# Confusion Matrix

svm_cm = confusion_matrix(y_test,svm_prediction)
sns.heatmap(svm_cm,annot=True,fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix")
plt.savefig("svm_confusion_matrix.jpeg")
plt.show()

# Hyperparameter tuning for SVM
param_grid = [
    {
        "kernel": ["linear"],
        "C": [0.1, 1, 10],
        "class_weight": [None, "balanced"]
    },
    {
        "kernel": ["rbf"],
        "C": [0.1, 1, 10],
        "gamma": ["scale", 0.01, 0.1],
        "class_weight": [None, "balanced"]
    }
]

svm_grid = GridSearchCV(
    estimator=SVC(
        probability=False,
        random_state=42
    ),
    param_grid=param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)

svm_grid.fit(X_train_scaled, y_train)
print("Best SVM Parameters:")
print(svm_grid.best_params_)
print("\nBest Cross-Validation ROC-AUC:")
print(svm_grid.best_score_)

# Optimized model
best_svm_model = SVC(
    **svm_grid.best_params_,
    probability=True,
    random_state=42
)
best_svm_model.fit(X_train_scaled, y_train)
best_svm_prediction = best_svm_model.predict(X_test_scaled)
best_svm_probability = best_svm_model.predict_proba(X_test_scaled)[:, 1]
print("\nOptimized SVM Accuracy:",accuracy_score(y_test, best_svm_prediction))
print("Optimized SVM Precision:",precision_score(y_test,best_svm_prediction,zero_division=0))
print("Optimized SVM Recall:",recall_score(y_test, best_svm_prediction))
print("Optimized SVM F1:",f1_score(y_test,best_svm_prediction,zero_division=0))
print("Optimized SVM ROC-AUC:",roc_auc_score(y_test, best_svm_probability))
print(classification_report(y_test,best_svm_prediction,zero_division=0))
# Confusion Matrix after tuning
best_svm_cm = confusion_matrix(y_test,best_svm_prediction)
sns.heatmap(best_svm_cm,annot=True,fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Optimized SVM Confusion Matrix")
plt.savefig("optimal_svm_confusion_matrix.jpeg")
plt.show()
# ROC Curve
svm_auc = roc_auc_score(y_test,best_svm_probability)
fpr,tpr,thresholds = roc_curve(y_test,best_svm_probability)
plt.figure(figsize=(8,6))
plt.plot(
    fpr,
    tpr,
    label=f"SVM (AUC={svm_auc:.3f})"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Optimized SVM")
plt.legend()
plt.savefig("roc_curve_optimised_svm")
plt.show()

# Threshold analysis
threshold_values = [0.1,0.2,0.3,0.4,0.5]
for threshold in threshold_values:
    svm_threshold_prediction = (
        best_svm_probability >= threshold
    ).astype(int)
    print("\nThreshold:", threshold)
    print("Recall:",recall_score(y_test,svm_threshold_prediction))
    print("F1:",f1_score(y_test,svm_threshold_prediction,zero_division=0))

# 2. K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",
    metric="minkowski"
)
knn_model.fit(
    X_train_scaled,
    y_train
)
knn_prediction = knn_model.predict(
    X_test_scaled
)
knn_probability = knn_model.predict_proba(
    X_test_scaled
)[:,1]
print("KNN Accuracy:",accuracy_score(y_test,knn_prediction))
print("KNN Precision:",precision_score(y_test,knn_prediction))
print("KNN Recall:",recall_score(y_test,knn_prediction))
print("KNN F1:",f1_score(y_test,knn_prediction))
print("KNN ROC-AUC:",roc_auc_score(y_test,knn_probability))
print(classification_report(y_test,knn_prediction))

# Confusion Matrix
knn_cm = confusion_matrix(
    y_test,
    knn_prediction
)
sns.heatmap(
    knn_cm,
    annot=True,
    fmt="d"
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN Confusion Matrix")
plt.savefig("Knn_confusion_matrix.jpeg")
plt.show()
# Hyperparameter tuning for KNN
param_grid = {
    "n_neighbors": [3,5,7,9,11,15,21],
    "weights": ["uniform","distance"],
    "metric": ["euclidean","manhattan","minkowski"]
}
knn_grid = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)
knn_grid.fit(
    X_train_scaled,
    y_train
)
print("Best KNN Parameters:")
print(knn_grid.best_params_)
print("\nBest Cross Validation ROC-AUC:")
print(knn_grid.best_score_)

# Best KNN model
best_knn_model = knn_grid.best_estimator_
# Evaluation after tuning
best_knn_prediction = best_knn_model.predict(X_test_scaled)
best_knn_probability = best_knn_model.predict_proba(X_test_scaled)[:,1]
print("\nOptimized KNN Accuracy:",accuracy_score(y_test,best_knn_prediction))
print("Optimized KNN Precision:",
      precision_score(
          y_test,
          best_knn_prediction,
          zero_division=0
      ))

print("Optimized KNN Recall:",
      recall_score(
          y_test,
          best_knn_prediction
      ))

print("Optimized KNN F1:",
      f1_score(
          y_test,
          best_knn_prediction,
          zero_division=0
      ))

print("Optimized KNN ROC-AUC:",
      roc_auc_score(
          y_test,
          best_knn_probability
      ))

print(
    classification_report(
        y_test,
        best_knn_prediction,
        zero_division=0
    )
)


# Confusion Matrix after tuning

best_knn_cm = confusion_matrix(
    y_test,
    best_knn_prediction
)


sns.heatmap(
    best_knn_cm,
    annot=True,
    fmt="d"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Optimized KNN Confusion Matrix")
plt.savefig("optimal_knn_confusion_matrix.jpeg")
plt.show()

# ROC Curve

knn_auc = roc_auc_score(
    y_test,
    best_knn_probability
)
fpr,tpr,thresholds = roc_curve(
    y_test,
    best_knn_probability
)
plt.figure(figsize=(8,6))
plt.plot(
    fpr,
    tpr,
    label=f"KNN (AUC={knn_auc:.3f})"
)
plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Optimized KNN")
plt.legend()
plt.savefig("optimised_knn_roc_curve.jpeg")
plt.show()

# Threshold analysis
threshold_values = [0.1,0.2,0.3,0.4,0.5]
best_threshold = 0
best_f1 = 0
for threshold in threshold_values:
    knn_predictiont = (
        best_knn_probability >= threshold
    ).astype(int)
    score = f1_score(
        y_test,
        knn_predictiont,
        zero_division=0
    )
    if score > best_f1:
        best_f1 = score
        best_threshold = threshold
print("Best Threshold:",best_threshold)
print("Best F1:",best_f1)
final_knn_prediction = (best_knn_probability >= best_threshold).astype(int)
final_knn_probability = best_knn_model.predict_proba(X_test_scaled)[:,1]
print(classification_report(y_test,final_knn_prediction))
# Confusion Matrix after tuning
best_knn_cm = confusion_matrix(y_test,final_knn_prediction)
sns.heatmap(best_knn_cm,annot=True,fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Optimized threshold KNN Confusion Matrix")
plt.savefig("optimal_threshold_knn_confusion_matrix.jpeg")
plt.show()

# Models Comparison
model_results = []
models = {
    "Logistic Regression": (
        y_test,
        logistic_pred,
        logistic_probability
    ),
    "Naive Bayes": (
        y_mnb_test,
        optimal_mnb_prediction,
        optimal_mnb_probability
    ),
    "SVM": (
        y_test,
        best_svm_prediction,
        best_svm_probability
    ),
    "KNN": (
        y_test,
        final_knn_prediction,
        final_knn_probability
    )
}
for model_name, (true, prediction, probability) in models.items():
    model_results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(
            true,
            prediction
        ),
        "Precision": precision_score(
            true,
            prediction,
            zero_division=0
        ),
        "Recall": recall_score(
            true,
            prediction,
            zero_division=0
        ),
        "F1-score": f1_score(
            true,
            prediction,
            zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            true,
            probability
        )
    })
comparison_table = pd.DataFrame(model_results)
print(comparison_table)
comparison_table.sort_values(by="ROC-AUC",ascending=False)

# Visualize Comparison
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC"
]
comparison_table.plot(
    x="Model",
    y=metrics,
    kind="bar",
    figsize=(12,6)
)
plt.title("Comparison of Probablistic & Non-Probabilistic Algorithms for Stroke Prediction")
plt.ylabel("Score")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05,1))
plt.tight_layout()
plt.savefig("probabilistic_vs_non_probabilistic_algorithms.jpeg")
plt.show()

best_model = comparison_table.loc[
    comparison_table["ROC-AUC"].idxmax()
]
print("Best Model Based on ROC-AUC:")
print(best_model)
best_accuracy = comparison_table.loc[
    comparison_table["Accuracy"].idxmax()
]
print(best_accuracy)