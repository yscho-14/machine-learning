import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# 1. Data Loading

train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

train.shape, test.shape

# generate new features
train['NPK_sum'] = train['Nitrogen'] + train['Phosphorous'] + train['Potassium']
test['NPK_sum'] = test['Nitrogen'] + test['Phosphorous'] + test['Potassium']

train['N_P_ratio'] = train['Nitrogen'] / (train['Phosphorous'] + 1)
test['N_P_ratio'] = test['Nitrogen'] / (test['Phosphorous'] + 1)

# 2. Data Overview & Visualization
print("\n=== [Step 2] Data Overview ===")
print(train.head())
print("Target value counts:")
print(train['Fertilizer Name'].value_counts())

plt.figure(figsize=(10,4))
sns.countplot(y='Fertilizer Name', data=train, order=train['Fertilizer Name'].value_counts().index)
plt.title("Class Distribution of Target (Fertilizer Name)")
plt.show()

# 3. OneHot Encoding
from sklearn.preprocessing import OneHotEncoder
cat_cols = ['Soil Type', 'Crop Type']
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_cat = ohe.fit_transform(train[cat_cols])
X_test_cat = ohe.transform(test[cat_cols])
num_cols = [col for col in train.columns if col not in ['id', 'Fertilizer Name'] + cat_cols]
X_num = train[num_cols].values
X_test_num = test[num_cols].values
X_all = np.hstack([X_num, X_cat])
X_test_all = np.hstack([X_test_num, X_test_cat])

features = [col for col in train.columns if col != 'Fertilizer Name']
target = 'Fertilizer Name'

# 4. H2O Initialization and Data Conversion
import h2o
from h2o.automl import H2OAutoML

print("\n=== [Step 4] H2O Initialization ===")
h2o.init()

train_h2o = h2o.H2OFrame(train)
test_h2o = h2o.H2OFrame(test)

# Set target as factor for classification
train_h2o[target] = train_h2o[target].asfactor()

# 5. AutoML Training
print("\n=== [Step 5] H2O AutoML Training ===")
aml = H2OAutoML(
    max_runtime_secs=900,   
    nfolds=5,
    seed=42,
    sort_metric="mean_per_class_error"
)
aml.train(x=features, y=target, training_frame=train_h2o)

# Install required libraries
!pip install polars pyarrow

# 6. Leaderboard Visualization
print("\n=== [Step 6] AutoML Leaderboard ===")
lb = aml.leaderboard.as_data_frame(use_multi_thread=True)
print(lb.head(10))

plt.figure(figsize=(6,2))
sns.barplot(x='mean_per_class_error', y='model_id', data=lb.head(10), palette="viridis")
plt.title("Top 10 AutoML Models (mean_per_class_error)")
plt.xlabel("Mean Per Class Error")
plt.ylabel("Model ID")
plt.show()

# 7. Validation Set Prediction & Evaluation
print("\n=== [Step 7] Validation Prediction & Evaluation ===")
from sklearn.model_selection import train_test_split

y = train['Fertilizer Name']
X_train, X_val, y_train, y_val = train_test_split(X_all, y, test_size=0.2, random_state=42, stratify=y)

# Convert validation set to H2OFrame
val_df = X_val.copy()
val_df[target] = y_val
val_h2o = h2o.H2OFrame(val_df)
val_h2o[target] = val_h2o[target].asfactor()

# Predict on validation set
val_pred = aml.leader.predict(val_h2o).as_data_frame()
val_true = y_val.values

# Top-3 prediction extraction
val_pred_proba = val_pred.iloc[:,1:].values  # skip first column (predicted label)
val_pred_top3_idx = np.argsort(-val_pred_proba, axis=1)[:,:3]
val_pred_top3_label = le_fert.inverse_transform(val_pred_top3_idx.ravel()).reshape(val_pred_top3_idx.shape)
val_true_label = le_fert.inverse_transform(val_true)

# Show some prediction examples
print("\n[Sample Top-3 Predictions vs True Label]")
for i in range(10):
    print(f"Predicted Top-3: {val_pred_top3_label[i]}, True: {val_true_label[i]}")

# Top-3 accuracy calculation
top3_correct = [val_true_label[i] in val_pred_top3_label[i] for i in range(len(val_true_label))]
top3_acc = np.mean(top3_correct)
print(f"\nTop-3 Validation Accuracy: {top3_acc:.4f}")

# 8. Prediction Probability Visualization
print("\n=== [Step 8] Prediction Probability Visualization ===")
plt.figure(figsize=(6,3))
sns.histplot(np.max(val_pred_proba, axis=1), bins=30, kde=True)
plt.title("Histogram of Max Predicted Probability (Validation Set)")
plt.xlabel("Max Predicted Probability")
plt.ylabel("Frequency")
plt.show()

# 9. Test Set Prediction and Submission
print("\n=== [Step 9] Test Prediction & Submission File ===")
test_pred = aml.leader.predict(test_h2o).as_data_frame()
test_pred_proba = test_pred.iloc[:,1:].values
test_pred_top3_idx = np.argsort(-test_pred_proba, axis=1)[:,:3]
test_pred_top3_label = le_fert.inverse_transform(test_pred_top3_idx.ravel()).reshape(test_pred_top3_idx.shape)

submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': [' '.join(row) for row in test_pred_top3_label]
})
submission.to_csv('submission.csv', index=False)
print(submission.head())

# 10. Summary Visualization: Top-1 Prediction Distribution
print("\n=== [Step 10] Top-1 Prediction Distribution on Test Set ===")
top1_pred_idx = np.argmax(test_pred_proba, axis=1)
top1_pred_label = le_fert.inverse_transform(top1_pred_idx)
plt.figure(figsize=(10,4))
sns.countplot(y=top1_pred_label, order=pd.Series(top1_pred_label).value_counts().index)
plt.title("Top-1 Predicted Fertilizer Distribution (Test Set)")
plt.xlabel("Count")
plt.ylabel("Fertilizer Name")
plt.show()
