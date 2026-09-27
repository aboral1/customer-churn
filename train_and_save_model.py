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
