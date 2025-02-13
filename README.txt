# Diabetes Prediction Using Machine Learning

A comprehensive machine learning project that implements multiple classification algorithms to predict diabetes based on diagnostic measurements. The models achieve accuracies ranging from 73% to 76%.

## Models and Performance

| Model | Accuracy | ROC-AUC Score |
|-------|----------|---------------|
| Logistic Regression | 75.33% | 0.833 |
| Decision Tree | 73.67% | 0.804 |
| Neural Network | 76.33% | - |
| Voting Classifier | 74.67% | 0.845 |

## Features Used
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

## Dependencies
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
import tensorflow as tf
```

## Data Preprocessing
1. Missing value imputation
2. Feature scaling using StandardScaler
3. Class imbalance handling with SMOTE
4. Train-test split (70-30)

## Model Architecture
- **Logistic Regression**: Base classifier
- **Decision Tree**: Max depth=5, min_samples_split=5
- **Neural Network**: 3 layers (16-8-1) with ReLU activation
- **Voting Classifier**: Soft voting with Logistic Regression and Decision Tree

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

## Key Findings
1. Glucose, BMI, and Insulin are the most significant predictors
2. PCA shows clear separation between classes
3. Neural Network achieves highest accuracy at 76.33%
4. Ensemble methods provide robust performance

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
# Run the main script
python diabetes_prediction.py
```

## Future Improvements
- Hyperparameter optimization
- Feature engineering
- Additional model architectures
- Cross-validation implementation
- Model deployment pipeline
