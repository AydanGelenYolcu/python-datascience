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
print(data.info())

#null values
print("null values: \n", data.isnull().sum())

data.dropna(subset=['default'], inplace=True)
#controlled
print(data.isnull().sum()) #succesful

#My data fell down to 700 from 1150
print(data.describe())


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


# create a copy of the original data to preserve it
data_engineered = data.copy()


# Feature Engineering
# Feature Engineering - create_features
def create_features(df):
    df = df.copy()
    
    # Create financial ratios
    df['total_debt'] = df['creddebt'] + df['othdebt']
    df['debt_to_income'] = df['total_debt'] / (df['income'] + 1e-10)
    df['credit_utilization'] = df['creddebt'] / (df['total_debt'] + 1e-10)
    df['monthly_income'] = df['income'] / 12
    df['debt_service_ratio'] = df['debtinc'] / 100
    
    # Create interaction features
    df['income_employ_interaction'] = df['income'] * df['employ']
    df['age_income_ratio'] = df['age'] / (df['income'] + 1e-10)
    
    # Create age groups
    df['age_group'] = pd.cut(df['age'], 
                             bins=[0, 25, 35, 45, 100], 
                             labels=['young', 'middle', 'mature', 'senior'])
    
    # Create income groups
    df['income_group'] = pd.cut(df['income'], 
                                bins=[0, 30, 60, 100, float('inf')], 
                                labels=['low', 'medium', 'high', 'very_high'])
    
    # Create debt ratio groups
    df['debt_ratio_group'] = pd.cut(df['debtinc'], 
                                    bins=[0, 10, 20, 30, float('inf')], 
                                    labels=['low', 'medium', 'high', 'very_high'])
    
    # One-hot encoding for categorical variables
    df = pd.get_dummies(df, columns=['age_group', 'income_group', 'debt_ratio_group'])
    
    return df

# 1. Age groups for age/default ratio graph
def create_age_groups_default(age):
    if age < 25:
        return '20-24'
    elif age < 30:
        return '25-29'
    elif age < 35:
        return '30-34'
    elif age < 40:
        return '35-39'
    elif age < 45:
        return '40-44'
    elif age < 50:
        return '45-49'
    else:
        return '50+'

# create a temporary column to generate and visualize the age groups
data_engineered['age_group2'] = data_engineered['age'].apply(create_age_groups_default)

# age and default ratio graph
plt.figure(figsize=(12, 6))
sns.barplot(x='age_group2', y='default', data=data_engineered, ci=None)
plt.title('Default Ratios - Age Groups')
plt.xlabel('Age group')
plt.ylabel('Default ratios')
plt.xticks(rotation=45)
plt.show()
# Feature engineering
data_engineered = create_features(data)

# dependent and independent variables
X = data_engineered.drop(columns=['default'])  # independent variables
y = data_engineered['default']  # target, dependent variable

#spliting datas as train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# prediction
y_pred = log_reg.predict(X_test_scaled)

# Model Performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ROC curve and auc
y_pred_prob = log_reg.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# ROC curve graph
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Calculate z-scores for numerical columns
numerical_cols = ['age', 'income', 'debtinc', 'creddebt', 'othdebt']
for col in numerical_cols:
    data[f'{col}_zscore'] = zscore(data[col])

# Calculate risk score
def calculate_risk_score(row):
    weights = {
        'debtinc': 0.35,
        'creddebt': 0.25,
        'othdebt': 0.20,
        'income': -0.15,
        'age': -0.05
    }
    score = 0
    for feature, weight in weights.items():
        normalized_value = (row[f'{feature}_zscore'] + 3) / 6
        score += normalized_value * weight
    return max(min(score, 1), 0)

# Add risk score before categorization
data['risk_score'] = data.apply(calculate_risk_score, axis=1)

def risk_category(score):
    if score < 0.2: return 'Very Low Risk'
    elif score < 0.4: return 'Low Risk'
    elif score < 0.6: return 'Medium Risk'
    elif score < 0.8: return 'High Risk'
    else: return 'Very High Risk'

data['risk_category'] = data['risk_score'].apply(risk_category)

# Continue with your existing visualizations and model training

data['risk_category'] = data['risk_score'].apply(risk_category)

# Visualize risk distribution
plt.figure(figsize=(12, 6))

# Risk score distribution
plt.subplot(1, 2, 1)
sns.histplot(data['risk_score'], bins=30, kde=True)
plt.title('Risk Score Distribution')
plt.xlabel('Risk Score')
plt.ylabel('Count')

# Risk categories
plt.subplot(1, 2, 2)
risk_order = ['Very Low Risk', 'Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
sns.countplot(data=data, x='risk_category', order=risk_order)
plt.xticks(rotation=45)
plt.title('Risk Category Distribution')
plt.tight_layout()
plt.show()

# Visualize feature importance with risk metrics
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': abs(log_reg.coef_[0])
})
feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Top 10 Most Important Features (including Risk Metrics)')
plt.tight_layout()
plt.show()
