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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f' Training set: {X_train.shape[0]} samples')
print(f' Test set: {X_test.shape[0]} samples')

# Hyperparameter tuning with GridSearch
print('\n5. Training Random Forest with GridSearch CV...')
print(' This may take a while...')

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

rf_tuned = RandomForestClassifier(random_state=42)

grid_search_rf = GridSearchCV(
    estimator=rf_tuned,
    param_grid=param_grid_rf,
    cv=5,
    n_jobs=-1,
    verbose=1,
    scoring='f1'
)

grid_search_rf.fit(X_train, y_train)

# Get the best model
best_churn_model = grid_search_rf.best_estimator_

print(f'\n Best parameters: {grid_search_rf.best_params_}')
print(f' Best CV F1 score: {grid_search_rf.best_score_:.4f}')

# Evaluate on test set
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

y_pred = best_churn_model.predict(X_test)
y_proba = best_churn_model.predict_proba(X_test)[:, 1]

print('\n6. Model Performance on Test set:')
print(f' Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f' Precision: {precision_score(y_test, y_pred):.4f}')
print(f' Recall: {recall_score(y_test, y_pred):.4f}')
print(f' F1 Score: {f1_score(y_test, y_pred):.4f}')
print(f' ROC AUC: {roc_auc_score(y_test, y_proba):.4f}')

# Save the model and encoders
print('\n7. Saving the model and encoders...')
joblib.dump(best_churn_model, 'model/best_churn_model.pkl')
joblib.dump(le_geography, 'model/le_geography.pkl')
joblib.dump(le_gender, 'model/le_gender.pkl')

# Save feature names
feature_names = X.columns.tolist()
with open('model/feature_names.json', 'w') as f:
    json.dump(feature_names, f)

# Save label encoder classes for reference
label_encoders_info = {
    'geography_classes': le_geography.classes_.tolist(),
    'gender_classes': le_gender.classes_.tolist(),
    'geography_mapping': dict(zip(le_geography.classes_, le_geography.transform(le_geography.classes_).tolist())),
    'gender_mapping': dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_).tolist()))
}

with open('model/label_encoder_info.json', 'w') as f:
    json.dump(label_encoders_info, f, indent=4)

print(' ✅ Model saved: model/best_churn_model.pkl')
print(' ✅ Geography encoder saved: model/le_geography.pkl')
print(' ✅ Gender encoder saved: model/le_gender.pkl')
print(' ✅ Feature names saved: model/feature_names.json')
print(' ✅ Label encoder info saved: model/label_encoder_info.json')

print('\n' + '=' * 70)
print('Model Training and Saving Completed Successfully!')
print('=' * 70)
print('\n You can now use the FASTAPI application to serve predictions using the trained model!')
