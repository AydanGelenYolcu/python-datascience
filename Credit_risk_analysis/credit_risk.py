import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc

#my dataset link = https://www.kaggle.com/datasets/atulmittal199174/credit-risk-analysis-for-extending-bank-loans
data = pd.read_csv("bankloans.csv")
print(data.head())
"""
age  ed  employ  address  income  debtinc   creddebt   othdebt  default
0   41   3      17       12     176      9.3  11.359392  5.008608      1.0
1   27   1      10        6      31     17.3   1.362202  4.000798      0.0
2   40   1      15       14      55      5.5   0.856075  2.168925      0.0
3   41   1      15       14     120      2.9   2.658720  0.821280      0.0
4   24   2       2        0      28     17.3   1.787436  3.056564      1.0"""


#null values
print("null values: \n", data.isnull().sum())

""" output:null values: 
 age           0
ed            0
employ        0
address       0
income        0
debtinc       0
creddebt      0
othdebt       0
default     450
dtype: int64"""

print(data.describe())
"""output age           ed       employ      address       income  \
 count  1150.000000  1150.000000  1150.000000  1150.000000  1150.000000   
 mean     35.235652     1.695652     8.781739     8.485217    47.982609   
 std       8.089961     0.927051     6.914762     6.977725    40.508814   
 min      20.000000     1.000000     0.000000     0.000000    13.000000   
 25%      29.000000     1.000000     3.000000     3.000000    24.000000   
 50%      35.000000     1.000000     7.000000     7.000000    36.000000   
 75%      41.000000     2.000000    13.000000    12.000000    56.750000   
 max      56.000000     5.000000    33.000000    34.000000   446.000000   
 
            debtinc     creddebt      othdebt     default  
 count  1150.000000  1150.000000  1150.000000  700.000000  
 mean     10.063391     1.605111     3.103844    0.261429  
 std       6.584288     2.135967     3.529273    0.439727  
 min       0.100000     0.011696     0.045584    0.000000  
 25%       5.200000     0.415584     1.047996    0.000000  
 50%       8.750000     0.899130     2.038053    0.000000  
 75%      13.600000     1.898820     3.864960    1.000000  
 max      41.300000    20.561310    35.197500    1.000000  )
 The age ranges from 20 to 56.
The debt-to-income ratio (debtinc) has a mean of around 10%, but it goes as high as 41.3%.
Credit debt and other debt have a wide range of values, 
indicating diverse debt levels among borrowers."""   

# handle missing values in the 'default' column with filling 0
data['default'].fillna(0, inplace=True)
#controlled
print(data.isnull().sum()) #succesful

# Set up the plotting environment
plt.figure(figsize=(15, 10))

# Plot distribution for Age
plt.subplot(2, 2, 1)
sns.histplot(data['age'], kde=True, color='skyblue')
plt.title('Age Distribution')

# Plot distribution for Income
plt.subplot(2, 2, 2)
sns.histplot(data['income'], kde=True, color='orange')
plt.title('Income Distribution')

# Plot distribution for Debt-to-Income Ratio (debtinc)
plt.subplot(2, 2, 3)
sns.histplot(data['debtinc'], kde=True, color='green')
plt.title('Debt-to-Income Ratio Distribution')

# Plot distribution for Default (Target Variable)
plt.subplot(2, 2, 4)
sns.countplot(x='default', data=data, palette='Set2')
plt.title('Default Count')

plt.tight_layout()
plt.show()
numerical_features = ['age', 'ed', 'employ', 'address', 'income', 'debtinc', 'creddebt', 'othdebt']

#SCALER
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[numerical_features])
X_scaled_df = pd.DataFrame(X_scaled, columns=numerical_features)
X_scaled_df.head()

#IS THERE OUTLIERS?
# Calculate Z-scores for all numerical columns
z_scores = np.abs(zscore(data[numerical_features]))

# Identify outliers (Z-score > 3)
outliers = (z_scores > 3).all(axis=1)
print(f"Number of outliers: {np.sum(outliers)}")
#no outliers

# Features (X) and target (y)
X = data.drop(columns=['default'])  # All columns except 'default'
y = data['default']  # Target column (default)

"""In credit risk analysis, the primary target variable is "default," 
indicating whether a borrower will fail to repay a loan.
This binary classification enables the training of predictive models 
that analyze customer data to assess their default risk.
These predictions are crucial for informed decision-making in loan approvals, 
risk management, and setting appropriate loan terms to minimize financial losses."""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(random_state=42)


model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)


accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)


print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", conf_matrix)

"""Accuracy: 0.808695652173913
Classification Report:
               precision    recall  f1-score   support

         0.0       0.84      0.95      0.89       192
         1.0       0.29      0.11      0.15        38

    accuracy                           0.81       230
   macro avg       0.56      0.53      0.52       230
weighted avg       0.75      0.81      0.77       230

Confusion Matrix:
 [[182  10]
 [ 34   4]]"""
 
 # 5. ROC eğrisini çizme
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# 6. Eğriyi çizme
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Rastgele tahmin çizgisi
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
 #RESULT: AUC 0.53 model is not better than random
 #second model with logistic regression, using class weight for sesivity
# Initialize the model with class weights (balanced) becaused even %80 accuracy
#first model is bad at detecting 1s which is default
model = LogisticRegression(class_weight='balanced', random_state=42)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
"""Accuracy: 0.6652173913043479
Classification Report:
               precision    recall  f1-score   support

         0.0       0.91      0.66      0.77       192
         1.0       0.29      0.68      0.40        38

    accuracy                           0.67       230
   macro avg       0.60      0.67      0.59       230
weighted avg       0.81      0.67      0.71       230

Confusion Matrix:
 [[127  65]
 [ 12  26]]"""
 # 5. ROC eğrisini çizme
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# 6. Eğriyi çizme
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Rastgele tahmin çizgisi
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
#Auc 0.67 nearly good
""" Results If identifying positive class instances (class 1) is more important, 
the 2nd model is better(finding customers with default). 
If classifying the negative class (class 0) is more important
(which means giving credits to customers, customers with no default), 
and it is okay with missing some positive class instances, then the 1st model is better."""
