# 🧠 Oral Cancer Prediction using Machine Learning

This project is a complete machine learning pipeline built to predict oral cancer diagnosis (`Yes`/`No`) using clinical, lifestyle, and demographic features. The model leverages a Gradient Boosting Classifier, with preprocessing, feature selection, and training steps built using `scikit-learn` pipelines.

---

## 📂 Dataset

The dataset used in this project contains **84,922 patient records** with 24 columns including:
- Demographic features (e.g., Age, Gender, Country)
- Lifestyle habits (e.g., Tobacco Use, Alcohol Consumption, Diet)
- Clinical indicators (e.g., Tumor Size, Cancer Stage)
- Outcome label: `Oral Cancer (Diagnosis)` - `Yes` or `No`

> **Note**: Sensitive or non-predictive fields like `Country` and `ID` were excluded from model training.

---

## 🧱 Project Pipeline

The ML pipeline consists of the following steps:

### 1. 🔍 Data Preprocessing
- **Missing Value Imputation**: Used `SimpleImputer` with most frequent strategy for numerical fields.
- **One-Hot Encoding**: Applied on categorical features like Gender, Tobacco Use, etc. using `OneHotEncoder` with `drop='first'`.
- **Target Mapping**: `'Oral Cancer (Diagnosis)'` was mapped to binary values: `Yes → 1`, `No → 0`.

### 2. ✅ Feature Selection
- Used `SelectFromModel` with `GradientBoostingClassifier` to automatically select the most important features.

### 3. 🧠 Model Training
- Final model trained using `GradientBoostingClassifier` (n_estimators=100, max_depth=3).
- Data split using `train_test_split` with stratification to maintain class distribution.

---

## ⚙️ Technologies Used

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib / seaborn (for EDA and plots)

---

## 📊 Model Performance

You can evaluate the model using standard classification metrics like:

- Accuracy
- Confusion Matrix
- Precision / Recall
- AUC Score

*To evaluate, you can extend the notebook with:*
```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = pipeline.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
