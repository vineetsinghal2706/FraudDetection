import mlflow
import mlflow.sklearn
import json
import joblib
import sys
import io
from mlflow.tracking import MlflowClient

# Fix Windows stdout encoding issue
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# MLflow tracking server URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("CreditCard_Fraud_Detection_V1")

# Load the trained model
model_path = "model.pkl"
model = joblib.load(model_path)

# Load evaluation metrics
with open("metrics.json", "r") as f:
    metrics = json.load(f)

# Define test thresholds
test_thresholds = {
    'Accuracy': 0.60,
    'Precision': 0.60,
    'Recall': 0.60,
    'F1 Score': 0.60,
    'Matthews Corrcoef': 0.60
}

# Function to check if metrics pass thresholds
def tests_pass(metrics, thresholds):
    for metric, threshold in thresholds.items():
        value = metrics.get(metric)
        if value is None:
            print(f"⚠️ Metric '{metric}' not found in metrics.json")
            return False
        if value < threshold:
            print(f"❌ Test failed: {metric} = {value:.4f} < threshold {threshold}")
            return False
    return True

# Run tests before logging
if tests_pass(metrics, test_thresholds):
    print("✅ All tests passed. Proceeding with model logging and registration...")

    with mlflow.start_run(run_name="Model Logging") as run:
        # Log model artifact and register it
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="CreditCardFraudModel"
        )

        # Log metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        # Log files as artifacts (optional)
        mlflow.log_artifact("metrics.json")
        mlflow.log_artifact("model.pkl")

        print(f"\n✅ Model logged and registered in MLflow as 'CreditCardFraudModel'")
        print(f"   Run ID: {run.info.run_id}")
        print(f"🏃 View run Model Logging at: http://127.0.0.1:5000/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
        print(f"🧪 View experiment at: http://127.0.0.1:5000/#/experiments/{run.info.experiment_id}")

        # Use MlflowClient to get latest version info and add tags
        client = MlflowClient()
        model_name = "CreditCardFraudModel"

        # Get all versions of the model sorted by creation time (descending)
        versions = client.search_model_versions(f"name='{model_name}'")

        # Assume the latest version is the one you just registered
        latest_version = max(versions, key=lambda v: int(v.version))

        model_version = latest_version.version

        # Add tags for status and role
        client.set_model_version_tag(
            name=model_name,
            version=model_version,
            key="role",
            value="challenger"
        )

        client.set_model_version_tag(
            name=model_name,
            version=model_version,
            key="status",
            value="staging"
        )

        print(f"🚀 Model version {model_version} tagged as 'challenger' and status 'staging'")

else:
    print("❌ Model failed the evaluation tests and will NOT be logged or registered.")
