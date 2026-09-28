#============================================================
# IMPORT SECTION
#============================================================
from ast import main

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import joblib
import json
import pandas as pd
import numpy as np
import os

#============================================================
# APP INITIALIZATION
#============================================================

# Create the FASTAPI application instane
app = FastAPI(
    title = 'Customer Churn Prediction API',
    description = 'API to predict Customer Churn using Random Forest model',
    version = '1.0.0'
)

#============================================================
# LOAD MODEL AND ENCODERS
#============================================================

# define the path and where our related files are stored
model_path = 'model'

# Load the trained Random Forest model
model = joblib.load(os.path.join(model_path, 'best_churn_model.pkl'))

# Load the label encoders
# We need these to transform Geography and Gender inputs
geography_encoder = joblib.load(os.path.join(model_path, 'le_geography.pkl'))
gender_encoder = joblib.load(os.path.join(model_path, 'le_gender.pkl'))

# Load feature names
with open(os.path.join(model_path, 'feature_names.json'), 'r') as f:
    feature_names = json.load(f)

# Load encoder info for reference
with open(os.path.join(model_path, 'label_encoder_info.json'), 'r') as f:
    label_encoder_info = json.load(f)

#============================================================
# PYDANTIC MODELS (DATA VALIDATION SCHEMAS)
#============================================================

class CustomerData(BaseModel):
    """
    Input Model - Defines the structure of customer data we expect
    
    This class inherits from BaseModel, which gives it superpowers:
    - Automatic validation of all fields
    - Automatic conversion to correct types
    - Clear error messages for invalid data
    
    Each field uses Field() to add extra validation and documentation:
    - description: Explains what the field is for
    - ge: "greater than or equal" - minimum value
    - le: "less than or equal" - maximum value
    - example: Shows an example value in API docs
    """

    # Credit Score: Must be between 300 and 850
    CreditScore: int = Field(
        ..., # Required field
        description="Customer's credit score (between 300 and 850)",
        ge=300, # Must be >= 300
        le=850, # Must be <= 850
        example=650
    )

    # Geography: Must be one of the countries
    Geography: str = Field(
        ...,
        description="Customer's country of residence (e.g., France, Spain or Germany)",
        example="France")

    # Gender: Must be either Male or Female
    Gender: str = Field(
        ...,
        description="Customer's gender (Male or Female)",
        example="Male")

    # Age: Must be between 18 and 100
    Age: int = Field(
        ...,
        ge=18,
        le=100,
        description="Customer's age (between 18 and 100)",
        example=35)

    # Tenure: Years with the bank, must be between 0 and 10
    Tenure: int = Field(
        ...,
        ge=0,
        le=10,
        description="Number of years the customer has been with the bank (0-10)",
        example=5)

    # Balance: Account balance, must be non-negative
    Balance: float = Field(
        ...,
        ge=0.0,
        description="Customer's account balance (non-negative)",
        example=50000.00)

    # NumOfProducts: Must be between 1 and 4
    NumOfProducts: int = Field(
        ...,
        ge=1,
        le=4,
        description="Number of bank products the customer is using (1-4)",
        example=2)

    # HasCrCard: 0 or 1 (boolean as integer)
    HasCrCard: int = Field(
        ...,
        ge=0,
        le=1,
        description="Whether the customer has a credit card (1) or not (0)",
        example=1)

    # IsActiveMember: 0 or 1 (boolean as integer)
    IsActiveMember: int = Field(
        ...,
        ge=0,
        le=1,
        description="Whether the customer is an active member (1) or not (0)",
        example=1)

    # EstimatedSalary: Must be non-negative
    EstimatedSalary: float = Field(
        ...,
        ge=0,
        description="Customer's estimated salary (non-negative)",
        example=100000.00)

class PredictionResponse(BaseModel):
    """
    Output Model - Defines the structure of our prediction response
    
    This tells FastAPI what our API will return to the user.
    """

    # Will the customer churn? 1 for Yes, 0 for No
    prediction: int = Field(
        ...,
        description="Predicted churn status: 1 for Yes, 0 for No",
    )

    # How confident is the model? (Probability from 0 to 1)
    churn_probability: float = Field(
        ...,
        description="Probability of churn (between 0 and 1)",
    )

    # Human-readable interpretation
    risk_level: str = Field(
        ...,
        description="Risk level based on churn probability (Low, Medium, High)",
    )

class BatchPredictionRequest(BaseModel):
    """
    Batch Input Model - For predicting multiple customers at once
    
    This allows users to send multiple customer records in a single request.
    """

    # List of customer data for batch prediction
    customers: List[CustomerData] = Field(
        ...,
        description="List of customer data for batch prediction",
    )

    # List[CustomerData] means: A list where each item is a CustomerData object
    # This enables bulk predictions, which is more efficient than individual requests


class BatchPredictionResponse(BaseModel):
    """
    Batch Output Model - Returns predictions for multiple customers
    """

    # List of predictions for each customer in the batch
    predictions: List[Dict] = Field(
        ...,
        description="List of predictions for each customer in the batch",
    )

    total_customers: int = Field(
        ...,
        description="Total number of customers processed in the batch",
    )

    total_churn: int = Field(
        ...,
        description="Total number of customers predicted to churn in the batch",
    )

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def preprocess_input(customer_data: CustomerData) -> pd.DataFrame:
    """
    Preprocesses a single customer input for prediction.
    
    This function:
    1. Validates that categorical values are known
    2. Encodes categorical variables (Geography, Gender) to numbers
    3. Arranges features in the correct order
    4. Converts the input into a DataFrame suitable for the model

    Parameters:
    -----------
    customer_data : CustomerData
        The customer data input from the API request, validated by Pydantic.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with one row, ready for prediction by the trained model

    Raises:
    ------
    HTTPException
        If the Geography or Gender values are not recognized by the pre-fitted LabelEncoders.
    """

    # Step 1: Validate Geography
    # Check if the provided country is one we know about
    if customer_data.Geography not in geography_encoder.classes_:
        # If not, raise an HTTP Error with status code 400 (Bad Request)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Geography value: '{customer_data.Geography}'. "
                   f"Expected one of: {list(geography_encoder.classes_)}"
        )

    # Step 2: Validate Gender
    # Check if the provided gender is one we know about
    if customer_data.Gender not in gender_encoder.classes_:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Gender value: '{customer_data.Gender}'. "
                   f"Expected one of: {list(gender_encoder.classes_)}"
        )

    # Step 3: Encode categorical variables
    geography_encoded = geography_encoder.transform([customer_data.Geography])[0]
    gender_encoded = gender_encoder.transform([customer_data.Gender])[0]

    # Step 4: Create feature dictionary
    # Build a dictionary with all features in the correct order
    # Note: The order of features must match the order used during model training
    feature_dict = {
        'CreditScore': customer_data.CreditScore,
        'Geography': geography_encoded,
        'Gender': gender_encoded,
        'Age': customer_data.Age,
        'Tenure': customer_data.Tenure,
        'Balance': customer_data.Balance,
        'NumOfProducts': customer_data.NumOfProducts,
        'HasCrCard': customer_data.HasCrCard,
        'IsActiveMember': customer_data.IsActiveMember,
        'EstimatedSalary': customer_data.EstimatedSalary
    }

    # Step 5: Convert to DataFrame
    df = pd.DataFrame([feature_dict])

    # Step 6: Ensure the DataFrame has the correct feature order
    df = df[feature_names]

    return df

def get_risk_level(probability: float) -> str:
    """
    Converts churn probability to a risk level category

    This makes the prediction easier to understand for business users.

    Parameters:
    -----------
    probability : float
        The predicted probability of churn (between 0 and 1).
    
    Returns:
    -------
    str
        Risk level as a string: 'Low', 'Medium', or 'High'.
    """

    if probability < 0.3:
        return 'Low'
    elif probability < 0.6:
        return 'Medium'
    else:
        return 'High'

# ============================================================
# API ENDPOINTS (ROUTES)
# ============================================================

@app.get("/")
def root():
    """
    Root endpoint - Welcome message
    
    Returns:
    -------
    A dictionary that FASTAPI automatically converts to a JSON response.
    """

    return {
        "message": "Customer Churn Prediction API",
        "status": "Active",
        "model": "Random Forest (Tuned)",
        "endpoints": {
            "predict": "/predict - Single customer prediction",
            "batch_predict": "/batch-predict - Multiple customer predictions",
            "health": "/health - API health check",
            "model_info": "/model-info - More details"
        }
    }

@app.get("/health")
def health_check():
    """
    Health check endpoint - Returns API status
    
    This is useful for monitoring and ensuring the API is running.
    
    Returns:
    -------
    A simple error message.
    """

    return {
        "status": "Healthy",
        "message": "model is not None"
    }

@app.get("/model-info")
def model_info():
    """
    Model information endpoint
    
    Returns information about the trained model
    Useful for users who want to know what model they are using.
    
    Returns:
    -------
    A dictionary with model details.
    """

    return {
        "model_type": "Random Forest Classifier",
        "status": "Tuned with GridSearchCV",
        "features": feature_names,
        "geography_options": list(geography_encoder.classes_),
        "gender_options": list(gender_encoder.classes_),
        "geography_mapping": label_encoder_info['geography_mapping'],
        "gender_mapping": label_encoder_info['gender_mapping'],
        "performance_metrics": {
            "accuracy": "~0.86",
            "precision": "0.76",
            "recall": "0.44",
            "f1_score": "0.56",
            "roc_auc": "0.85"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    """
    Single customer prediction endpoint.
    Predicts whether a single customer will churn.
    
    This endpoint takes validated customer data, preprocesses it,
    and uses the trained Random Forest model to predict churn.

    Parameters:
    -----------
    customer : CustomerData
        The input customer data validated by Pydantic.

    Returns:
    -------
    PredictionResponse
        A structured response containing the prediction, probability, and risk level.

    Raises:
    ------
    HTTPException
        If there is an error during preprocessing or prediction.
    """

    try:
        # Step 1: Preprocess the input data
        # Convert customer data to the format the model expects
        input_df = preprocess_input(customer)

        # Step 2: Make prediction
        # predict() returns 0 or 1 (no churn or churn)
        prediction = model.predict(input_df)[0]

        # Step 3: Get prediction probability
        # predict_proba() returns probability in each class [prob_no_churn, prob_churn]
        # We take the second value [1] which is the probability of churning
        churn_probability = model.predict_proba(input_df)[0][1]

        # Step 4: Determine risk level
        risk_level = get_risk_level(churn_probability)

        # Step 5: Return the response
        # FASTAPI automatically converts this to JSON
        return {
            "prediction": int(prediction), # Convert numpy int to Python int
            "churn_probability": float(churn_probability), # Convert numpy float to Python float
            "risk level": risk_level
        }

    except HTTPException:
        # Re-raise HTTPExceptions (validation errors)
        raise

    except Exception as e:
        # Catch any other unexpected errors
        # Return a 500 Internal Server Error
        raise HTTPException(
            status_code= 500,
            detail= f'Prediction error: {str(e)}'
        )

@app.post("/batch-predict", response_model= BatchPredictionResponse)
def batch_predict_churn(batch_request: BatchPredictionRequest):
    """
    Batch prediction endpoint for multiple customers

    This endpoint allows predicting churn for multiple customers in a single request.
    This is more efficient than making individual requests for each customer.

    Parameters:
    -----------
    batch_request = BatchPredictionRequest
        contains a list of customer data objects

    Returns:
    --------
    BatchPredictionResponse
        contains predictions for all customers plus summary statistics

    Raises:
    -------
    HTTPException
        If there's an error during batch processing
    """

    try:
        # Initialize results list
        predictions = []
        churn_count = 0

        # Process each customer
        # enumerate gives us both the index and the customer data
        for idx, customer in enumerate(batch_request.customers):
            try:
                # Preprocess customer data
                input_df = preprocess_input(customer)

                # Make prediction
                prediction = model.predict(input_df)[0]
                churn_probability = model.predict_proba(input_df)[0][1]
                risk_level = get_risk_level(churn_probability)

                # Count churners
                if prediction == 1:
                    churn_count += 1

                # Add to results
                predictions.append({
                    "customer_index": idx, # Which customer in the batch
                    "prediction": int(prediction),
                    "churn_probability": float(churn_probability),
                    "risk_level": risk_level
                })

            except Exception as e:
                # If one customer fails, include the error but continue with others
                predictions.append({
                    "customer_index": idx,
                    "error": str(e) 
                })

        # Return batch results
        return {
            "predictions": predictions,
            "total_customers": len(batch_request.customers),
            "total_churn": churn_count
        }

    except Exception as e:
        raise HTTPException(
            status_code= 500,
            detail= f'Batch prediction error: {str(e)}'
        )

# ============================================================
# MAIN EXECUTION
# ============================================================

"""
This section only runs when you execute this file directly.
It won't run when FASTAPI imports this file.

To run this API:
---------------
In your terminal, navigate to the project directory and run:

    uvicorn main:app --reload

The API will start at: http://127.0.0.1:8000

Accessing the API:
------------------
1. Interactive docs: http://127.0.0.1:8000/docs (Swagger UI)
2. Alternative docs: http://127.0.0.1:8000/redoc (ReDoc)
3. API root: http://127.0.0.1:8000/

These documentation pages are automatically generated by FASTAPI!
You can test your API directly from the browser using these interfaces.
"""

if __name__ == '__main__':
    import uvicorn

    # Run the FASTAPI application
    # host = '0.0.0.0' makes it accessible to other computers on the network
    # port = 8000 is the port number
    # reload = True automatically restarts when the code is changed
    uvicorn.run(
        'main:app',
        host = '0.0.0.0',
        port = 8000,
        reload = True
    )