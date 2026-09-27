# Model training and saving script
# This script trains the best model from the EDA script and saves it for deployment

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import json

print("=" * 70)
print("Training and Saving the best model - Random Forest (Tuned)")
print("=" * 70)

# Load the dataset
print('\n1. loading dataset...')
df = pd.read_csv('data/churn.csv')
print(f"  Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Data cleaning - drop unnecessary columns
print('\n2. Cleaning data...')
df_clean = df.drop(columns= ['RowNumber', 'CustomerId', 'Surname'])
print(f'  Dropped unnecessary columne. New shape: {df_clean.shape}')

# Encode categorical variables
print('\n3. Encoding categorical variables...')
le_geography = LabelEncoder()
le_gender = LabelEncoder()

df_clean['Geography'] = le_geography.fit_transform(df_clean['Geography'])
df_clean['Gender'] = le_gender.fit_transform(df_clean['Gender'])

print(f' Geography mapping: {dict(zip(le_geography.classes_, le_geography.transform(le_geography.classes_)))}')
print(f' Gender mapping: {dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))}')

# Split features and target
print('\n4. Splitting features and target...')
X = df_clean.drop(columns=['Exited'])
y = df_clean['Exited']

print(y)


