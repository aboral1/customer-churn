# Customer Churn Prediction

An end-to-end customer-churn project for a bank. It includes exploratory data analysis (EDA), model comparison, a saved Random Forest model, and a FastAPI service for real-time single and batch predictions.

## What is included

- EDA of 10,000 banking customer records
- Data cleaning, categorical encoding, and train/test splitting
- Logistic Regression, Decision Tree, and Random Forest models
- Hyperparameter tuning with `GridSearchCV`
- Saved model and label-encoder artifacts for inference
- FastAPI endpoints for health checks, model metadata, and predictions

## Project structure

```text
Customer Churn/
├── data/
│   └── churn.csv                     # Source dataset
├── model/                            # Generated locally; not committed
│   ├── best_churn_model.pkl          # Trained Random Forest model
│   ├── le_geography.pkl              # Geography label encoder
│   ├── le_gender.pkl                 # Gender label encoder
│   ├── feature_names.json            # Model feature order
│   └── label_encoder_info.json       # Category mappings
├── customer_churn_eda.ipynb          # EDA and model-comparison notebook
├── train_and_save_model.py           # Recreates model artifacts
├── main.py                           # FastAPI application
├── requirements.txt                  # Notebook/EDA dependencies
├── requirements_api.txt              # API dependencies
├── API_GUIDE.md                      # API setup and endpoint guide
└── README.md
```

## Dataset

`data/churn.csv` contains 10,000 customer records. The model uses these input features:

| Feature | Description |
| --- | --- |
| `CreditScore` | Customer credit score |
| `Geography` | France, Germany, or Spain |
| `Gender` | Female or Male |
| `Age` | Customer age |
| `Tenure` | Years with the bank |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products |
| `HasCrCard` | Has a credit card (`0` or `1`) |
| `IsActiveMember` | Active-member flag (`0` or `1`) |
| `EstimatedSalary` | Estimated annual salary |

`RowNumber`, `CustomerId`, and `Surname` are removed before model training. `Exited` is the target: `1` means the customer churned and `0` means they did not.

## Setup

Use Python 3.9–3.12; Python 3.11 is a safe default for this pinned dependency set and the saved model artifacts. The project's current package pins do not provide wheels for Python 3.14. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows PowerShell
```

### Notebook / EDA environment

Install the notebook dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook customer_churn_eda.ipynb
```

Run the notebook sequentially to reproduce the analysis and model comparison.

### API environment

Install the API dependencies:

```bash
pip install -r requirements_api.txt
```

The `model/` directory is generated locally and is not included in the repository. Generate its required artifacts from the dataset before starting the API:

```bash
python train_and_save_model.py
```

The training script runs a 540-combination Random Forest grid over 5 folds, so it can take considerably longer than a few minutes depending on the machine.

`main.py` loads these artifacts during startup. It will not start until the training script has created all files in `model/`.

Start the development API:

```bash
uvicorn main:app --reload
```

Then open <http://127.0.0.1:8000/docs> for interactive API documentation. See [API_GUIDE.md](API_GUIDE.md) for request examples, response contracts, troubleshooting, and deployment guidance.

## Analysis workflow

```text
Raw customer data
  → cleaning and feature selection
  → label encoding of Geography and Gender
  → stratified 80/20 train-test split
  → baseline and tuned model evaluation
  → saved Random Forest model and encoders
  → FastAPI prediction service
```

The notebook includes exploratory analysis, visualizations, model evaluation, feature importance, and 10-fold cross-validation for selected models. The separate deployment training script uses 5-fold cross-validation and optimizes F1 score.

## Saved notebook results

The following values are the saved outputs from `customer_churn_eda.ipynb` on its held-out test set. They are not a claim that the figures will be identical after retraining.

| Model | Accuracy | Precision | Recall | F1 score | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8050 | 0.5859 | 0.1425 | 0.2292 | 0.7710 |
| Decision Tree (Default) | 0.7765 | 0.4533 | 0.4767 | 0.4647 | 0.6649 |
| Decision Tree (Tuned) | 0.8345 | 0.6145 | 0.5012 | 0.5521 | 0.7796 |
| Random Forest (Default) | **0.8650** | **0.7819** | 0.4668 | **0.5846** | 0.8467 |
| Random Forest (Tuned) | 0.8580 | 0.7639 | 0.4373 | 0.5562 | **0.8505** |

There is no single universal winner: the default Random Forest is strongest on the saved accuracy, precision, and F1 metrics, while the tuned Random Forest has the highest ROC AUC. Choose the deployment model and decision threshold based on the business cost of false positives versus false negatives.

## Key findings from the EDA

- Churn is imbalanced at roughly 20% of the data.
- Older customers, particularly middle-aged and older groups, show higher churn.
- Customers in Germany show a higher churn rate than customers in France or Spain.
- Female customers show higher churn in the saved analysis.
- Inactive customers and customers with three or four products are higher-risk segments.
- Age, balance, geography, customer activity, and product count are important model inputs.

These patterns are associations in this dataset, not causal conclusions. Validate them before using them for customer decisions.

## API notes

The API validates numeric ranges and allowed categorical values, then returns a prediction, churn probability, and risk classification. It supports:

- `GET /` — API metadata
- `GET /health` — basic liveness response
- `GET /model-info` — feature and encoder information
- `POST /predict` — one customer prediction
- `POST /batch-predict` — multiple customer predictions

The current `/predict` implementation has a known response-key mismatch: its response schema expects `risk_level`, while the route returns `risk level`. This can cause a valid prediction to return HTTP `500`. The intended response contract and current behavior are documented in [API_GUIDE.md](API_GUIDE.md).

## Production considerations

`uvicorn main:app --reload` is a development command, not a production deployment. In production, deploy a versioned application and model artifact behind HTTPS, with authentication, restricted CORS, request limits, logging/monitoring, health checks, and automatic restart behavior. See [API_GUIDE.md](API_GUIDE.md) for the production checklist.

## License

Educational project; available for educational and personal use.
