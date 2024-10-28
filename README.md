# Loan Approval Prediction

This project aims to predict the loan approval status based on various applicant features using machine learning algorithms. The model is built using Python and leverages libraries like Pandas, NumPy, Matplotlib, Seaborn, and Scikit-Learn for data manipulation, visualization, and model training.

## Getting Started

To get started with this project, clone the repository to your local machine:

```bash
git clone https://github.com/iBelalAhmad/Loan_Approval_Predictor.git
cd LoanApprovalPrediction
```
---

## Dataset
The dataset used in this project is LoanApprovalPrediction.csv, which contains information regarding loan applicants and whether their loan was approved.

# Steps
## Step 1: Import Libraries and Dataset
Initially, import the necessary libraries and the dataset.

```bash
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("LoanApprovalPrediction.csv")

```

## Step 2: Data Preprocessing and Visualization
- Dropping Irrelevant Columns: 
The Loan_ID column is dropped as it is a unique identifier and does not contribute to predictions.
- Visualizing Categorical Variables: 
Categorical variables are visualized using bar plots to understand their distribution.
- Encoding Categorical Variables: 
Label Encoding is applied to convert categorical variables into numerical values.
- Correlation Heatmap: 
A heatmap is generated to visualize the relationships between different features.
- Handling Missing Values: 
Missing values are filled with the mean of their respective columns.
- Feature Engineering: 
A new feature TotalIncome is created by combining ApplicantIncome and CoapplicantIncome.

## Step 3: Choosing Models and Training the Models
Four different machine learning models are initialized:

- KNeighborsClassifier
- RandomForestClassifier
- SVC (Support Vector Classifier)
- LogisticRegression
Each model is trained on the training data, and predictions are made.

## Step 4: Testing the Models on the Test Set
Predictions are made on the test set for each model.
The performance of each model is evaluated using appropriate metrics (e.g., accuracy, precision, recall).
## Step 5: Tuning
Feature scaling is applied using Standard Scaler to improve model performance.
Models are retrained and re-evaluated after scaling the features.
## Step 6: Cross-Validation
10-fold cross-validation is performed to assess the performance of each model across different splits of the data.
This helps in understanding the model's stability and generalization to unseen data.

### Conclusion
Based on the cross-validation scores, the Logistic Regression model provides the best performance with a mean accuracy of approximately 80.2%.
