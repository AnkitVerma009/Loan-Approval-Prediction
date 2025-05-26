# Loan-Approval-Prediction-Using-Machine-Learning
A machine learning project that predicts loan approval outcomes using logistic regression and random forest classifiers. The notebook demonstrates data exploration, preprocessing, handling class imbalance with SMOTE, and evaluating model performance using ROC-AUC and other metrics.

## Project Objective

Banks and financial institutions receive a high volume of loan applications. Manually evaluating each application is time-consuming and prone to inconsistency. This project demonstrates how machine learning can help make faster and fairer decisions using data-driven methods.

## Data Insights

The dataset used is imbalanced, with 76% of the applications being approved and 24% not approved. This imbalance was addressed using SMOTE (Synthetic Minority Over-sampling Technique) to improve model performance and ensure fairness.

Initial exploration helped identify missing values, distribution of income, credit scores, and approval trends. Several visualizations were created to better understand how different variables affect the approval outcome.

## Workflow

1. Load and explore the dataset
2. Handle missing values and clean data
3. Encode categorical variables and scale numerical features
4. Apply SMOTE to balance the dataset
5. Train and test the following models:
   - Logistic Regression
   - Random Forest Classifier
6. Evaluate performance using:
   - Accuracy
   - ROC-AUC score
   - Confusion matrix
   - Classification report

## Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- imbalanced-learn (SMOTE)
