# Customer Churn Prediction API Guide

This project serves a trained Random Forest churn model through FastAPI. It validates customer data, applies the label encoders saved during training, and produces a churn prediction and probability.

## Prerequisites

- Python 3.8 or later
- Run all commands from this project's root directory
- A `model/` directory containing the model, encoders, feature names, and encoder metadata

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows PowerShell
```

Install API dependencies:

```bash
pip install -r requirements_api.txt
```

This repository includes model artifacts in `model/`. If they are missing, generate them with:

```bash
python train_and_save_model.py
```

Training uses GridSearchCV and can take a while. In a deployed application, model artifacts should be trained, versioned, and tested before deployment—not trained each time the API starts.

## Run locally

Start the development server:

```bash
uvicorn main:app --reload
```

`--reload` is for local development only. It watches source files and restarts the server when they change.

Open one of these URLs once the server starts:

- API root: <http://127.0.0.1:8000/>
- Swagger UI: <http://127.0.0.1:8000/docs>
- ReDoc: <http://127.0.0.1:8000/redoc>

## Endpoints

### `GET /`

Returns API metadata and route names.

```json
{
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
```

### `GET /health`

Returns a simple process health response:

```json
{
  "status": "Healthy",
  "message": "model is not None"
}
```

The model is loaded when the application is imported, so the app cannot start if the required model files cannot be loaded. This endpoint is a basic liveness check; it does not perform a test prediction.

### `GET /model-info`

Returns the expected feature order, category options, label-encoder mappings, and the metrics currently stored in the code.

Valid categorical values are:

- `Geography`: `France`, `Germany`, or `Spain`
- `Gender`: `Female` or `Male`

### `POST /predict`

Sends one customer record for prediction.

```json
{
  "CreditScore": 700,
  "Geography": "France",
  "Gender": "Male",
  "Age": 35,
  "Tenure": 5,
  "Balance": 75000.0,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 100000.0
}
```

The intended successful response contract is:

```json
{
  "prediction": 0,
  "churn_probability": 0.23,
  "risk_level": "Low"
}
```

`prediction` is `0` for no churn and `1` for churn. `risk_level` is derived from the churn probability:

- `Low`: below 0.30
- `Medium`: 0.30 to below 0.60
- `High`: 0.60 or above

#### Current implementation note

The current `/predict` implementation returns a key named `risk level` (with a space), while its FastAPI response schema declares `risk_level` (with an underscore). Therefore, a valid prediction currently fails response validation and can return HTTP `500`. This guide documents the intended contract above; verify the endpoint after the implementation and schema use the same key.

### Input validation

| Field | Accepted values |
| --- | --- |
| `CreditScore` | Integer from 300 to 850 |
| `Age` | Integer from 18 to 100 |
| `Tenure` | Integer from 0 to 10 |
| `Balance`, `EstimatedSalary` | Number greater than or equal to 0 |
| `NumOfProducts` | Integer from 1 to 4 |
| `HasCrCard`, `IsActiveMember` | Integer `0` or `1` |

Missing fields, incorrect types, and invalid numeric ranges return HTTP `422`. Unknown geography or gender values return HTTP `400`.

### `POST /batch-predict`

Accepts multiple complete customer records in a `customers` array:

```json
{
  "customers": [
    {
      "CreditScore": 700,
      "Geography": "France",
      "Gender": "Male",
      "Age": 35,
      "Tenure": 5,
      "Balance": 75000.0,
      "NumOfProducts": 2,
      "HasCrCard": 1,
      "IsActiveMember": 1,
      "EstimatedSalary": 100000.0
    }
  ]
}
```

Each successful item in `predictions` contains `customer_index`, `prediction`, `churn_probability`, and `risk_level`; the response also includes `total_customers` and `total_churn`.

The current batch endpoint continues if an individual record fails during processing. It adds an `error` object for that record and still returns HTTP `200`; clients must therefore handle both prediction and error objects in the list.

## Testing

For manual testing, use Swagger UI:

1. Run `uvicorn main:app --reload`.
2. Open <http://127.0.0.1:8000/docs>.
3. Choose an endpoint, select **Try it out**, provide JSON, then select **Execute**.

You can also use curl:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 700,
    "Geography": "France",
    "Gender": "Male",
    "Age": 35,
    "Tenure": 5,
    "Balance": 75000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 100000.0
  }'
```

There is no `test_api.py` in this repository. Add automated tests using FastAPI's `TestClient` before deploying changes.

## Request flow

```text
JSON request
  → Pydantic validates fields and numeric limits
  → Geography and Gender are checked and label-encoded
  → Features are placed in their training-time order
  → Random Forest predicts class and churn probability
  → FastAPI serializes and validates the JSON response
```

## Production deployment

For a non-development process, use a command such as:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
```

This command alone is not a complete production deployment. A typical setup is:

```text
Client → HTTPS load balancer/reverse proxy → FastAPI/Uvicorn service → model artifact
                                             ├─ authentication and rate limits
                                             ├─ logs, metrics, and alerts
                                             └─ health/readiness checks
```

Before exposing the API publicly:

- Build and deploy a versioned, tested application and model artifact together.
- Configure settings and model paths through environment variables; avoid relying on the current working directory.
- Use a container platform, cloud service, or process manager to start the app on boot and restart it after crashes.
- Terminate HTTPS at a reverse proxy or cloud load balancer.
- Add authentication, authorization, and rate limiting for prediction routes.
- Restrict CORS to approved frontend origins; do not use `allow_origins=["*"]` for a private or authenticated API.
- Set a maximum batch size and request timeout.
- Log latency and outcomes, but do not log customer data unless it is necessary and appropriately protected.
- Select workers based on available CPU and memory; each worker may load its own copy of the model.
- Consider disabling or protecting `/docs` and `/redoc` in public environments.

See the official [FastAPI deployment documentation](https://fastapi.tiangolo.com/deployment/concepts/) for deployment concepts.

## Troubleshooting

| Problem | Resolution |
| --- | --- |
| `ModuleNotFoundError` | Activate the virtual environment and run `pip install -r requirements_api.txt`. |
| Model file not found | Run from the project root and check that all artifacts exist under `model/`. |
| Port 8000 is in use | Run `uvicorn main:app --reload --port 8001`. |
| HTTP 422 | Check the response details for a missing field, type mismatch, or numeric constraint. |
| HTTP 400 for a category | Use a category listed by `/model-info`. |
| HTTP 500 from `/predict` | See the current implementation note under the `/predict` endpoint. |
