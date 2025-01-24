# Credit Risk Analysis Project

## Overview
This project focuses on credit risk analysis, specifically analyzing bank loans and predicting loan defaults. The dataset used includes various financial and demographic features of borrowers. The project applies multiple machine learning techniques and risk scoring methodologies to assess borrowers' risk levels and predict default rates.

## Dataset
The dataset used in this project is obtained from [Kaggle](https://www.kaggle.com/datasets/atulmittal199174/credit-risk-analysis-for-extending-bank-loans).

- **Data Columns**:
  - age: Age of the borrower
  - income: Income level of the borrower
  - debtinc: Debt-to-income ratio
  - creddebt: Credit card debt
  - othdebt: Other types of debt
  - default: Whether the borrower has defaulted on loans (0 = No, 1 = Yes)

## Feature Engineering
Several new features were created to enhance the prediction model:
- **total_debt**: Sum of credit card and other debt.
- **debt_to_income**: Ratio of total debt to income.
- **credit_utilization**: Ratio of credit card debt to total debt.
- **monthly_income**: Income divided by 12 to represent monthly income.
- **debt_service_ratio**: Debt-to-income ratio divided by 100.
- **income_employ_interaction**: Interaction feature between income and years of employment.
- **age_income_ratio**: Age divided by income.
- **age groups**: Grouped into categories: 'young', 'middle', 'mature', 'senior'.
- **income groups**: Grouped income into low, medium, high, and very high.
- **debt ratio groups**: Grouped debt-to-income ratios into low, medium, high, very high.

## Model Training
A logistic regression model is used to predict loan defaults, with train-test splitting to validate model performance. The features are scaled using `StandardScaler`, and key metrics such as accuracy, classification report, confusion matrix, and ROC-AUC scores are analyzed.

## Risk Scoring System
Z-scores are calculated for numerical columns such as age, income, debtinc, creddebt, and othdebt. Based on the normalized z-scores, a custom risk score is generated for each borrower. Weights are assigned to different features to compute the risk score:

- **debtinc**: 0.35
- **creddebt**: 0.25
- **othdebt**: 0.20
- **income**: -0.15
- **age**: -0.05

Risk scores are categorized into five risk categories: Very Low Risk, Low Risk, Medium Risk, High Risk, and Very High Risk.

## Key Visualizations
- Age, Income, and Debt-to-Income Ratio Distributions.
- Default count across different age groups.
- Risk score and risk category distributions.
- Top 10 most important features affecting loan default prediction.
Insights from the Data:

**Age Distribution**: The age distribution shows a concentration of borrowers in the 25-40 age range, with a peak around 30 years old. Older borrowers over 45 make up a smaller portion of the dataset.
**Income Distribution**: The income distribution is highly skewed, with a long tail of high-income borrowers. The majority of borrowers have incomes under 100.
**Debt-to-Income Ratio Distribution**: The debt-to-income ratio shows a bimodal distribution, with peaks around 10% and 30%. This suggests there are two distinct segments of borrowers with different debt levels.
**Default Rates by Age Group**: The default rates are highest for the youngest (20-24) and oldest (50+) age groups, with the middle-aged groups having lower default rates.
![image](https://github.com/user-attachments/assets/3423e34c-34e0-4796-b40d-f9afff0fddc0)
![image](https://github.com/user-attachments/assets/3852664f-a504-466d-835c-2289a2f81c3a)

## Insights from the Model:

**Model Accuracy** : The logistic regression model achieves an accuracy of 88% on the test set, which is a strong performance.
**Feature Importance**: The most important features for predicting default include debt-related metrics like total debt, credit debt, and debt-to-income ratio. Income and age also have significant predictive power.
**Risk Score Distribution**: The risk score distribution shows a bimodal pattern, with a large portion of borrowers having low risk scores and a smaller group with high risk scores.
**Risk Category Distribution**: Categorizing borrowers into risk groups, the majority fall into the "Low Risk" and "Medium Risk" categories, with smaller portions in the "Very Low", "High", and "Very High" risk groups.
**ROC Curve**: The ROC curve has an AUC of 0.88, indicating the model has good discriminative power in separating defaulting and non-defaulting borrowers.
![image](https://github.com/user-attachments/assets/6f2991d8-4a8e-43ec-bf3a-724869c536e6)
![image](https://github.com/user-attachments/assets/3d903afa-a42b-42c3-abbe-6c639a98caac)
![image](https://github.com/user-attachments/assets/fb449c1d-fa53-49ad-b7a1-7fe60febf549)
![image](https://github.com/user-attachments/assets/ce5a588d-1c66-4a7c-a51d-e3ed53f039ec)

## Conclusion
1. The model achieves 88% accuracy in predicting default, indicating strong performance.
2. Debt-related metrics like total debt, credit debt, and debt-to-income ratio are the most important features.
3. Age and income also have significant predictive power, with highest defaults for youngest and oldest borrowers.
4. The risk score distribution is bimodal, with a large low-risk segment and smaller high-risk group.
5. The majority of borrowers fall into low or medium risk categories, with smaller portions in very low, high, and very high risk.

These insights suggest the model can effectively stratify borrowers by default risk using a combination of demographic, financial, and debt-related factors. This could help inform credit policies and underwriting decisions to manage risk. Overall, the analysis provides valuable input for improving the bank's lending practices.
