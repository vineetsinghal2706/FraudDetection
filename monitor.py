import os
import pandas as pd
import snowflake.connector
from evidently.core.report import Report
from evidently.presets import DataDriftPreset, ClassificationPreset
from evidently import Dataset, DataDefinition
import json
import io
import sys
from evidently import BinaryClassification
import pickle

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')  
# Load Snowflake credentials from environment variables

import mlflow
import snowflake.connector
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)



# Load config from environment
account = os.getenv('SNOWFLAKE_ACCOUNT')
user = os.getenv('SNOWFLAKE_USER')
password = os.getenv('SNOWFLAKE_PASSWORD')
warehouse = os.getenv('SNOWFLAKE_WAREHOUSE')
database = os.getenv('SNOWFLAKE_DATABASE')
schema = os.getenv('SNOWFLAKE_SCHEMA')

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI",'http://127.0.0.1:5000'))
mlflow.set_experiment("Monitoring_Experiments_V1")

def fetch_from_snowflake(query):
    conn = snowflake.connector.connect(
        user=user, password=password,
        account=account, warehouse=warehouse,
        database=database, schema=schema
    )
    df = conn.cursor().execute(query).fetch_pandas_all()
    conn.close()
    return df

def load_champion_model():
    model_path = "champion_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found in the current directory.")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("✅ Loaded champion model from local champion_model.pkl")
    return model

def calc_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1_Score": f1_score(y_true, y_pred),
        "MatthewsCorrcoef": matthews_corrcoef(y_true, y_pred),
    }

def main():
    model = load_champion_model()

    ref = fetch_from_snowflake("SELECT * FROM CREDITCARD_REFERENCE.PUBLIC.CREDITCARD_REFERENCE")
    cur = fetch_from_snowflake("SELECT * FROM CREDITCARD.PUBLIC.CREDITCARD_BATCH_INPUTS")
    target = "CLASS"

    # Only use original feature columns for prediction and monitoring
    feature_cols = [col for col in ref.columns if col not in ['ID', 'CLASS', 'PREDICTION', 'PREDICTION_PROB']]
    ref[feature_cols] = ref[feature_cols].apply(pd.to_numeric, errors='coerce')
    cur[feature_cols] = cur[feature_cols].apply(pd.to_numeric, errors='coerce')

    ref["prediction"] = model.predict(ref[feature_cols])
    cur["prediction"] = model.predict(cur[feature_cols])

    # dd = DataDefinition(
    #     numerical_columns=feature_cols,
    #     categorical_columns=None
    # ) 
    dd = DataDefinition(
    classification=[BinaryClassification(
        target="CLASS",
        prediction_labels="prediction")],
    categorical_columns=["CLASS", "prediction"])


    ds_ref = Dataset.from_pandas(ref, data_definition=dd)
    ds_cur = Dataset.from_pandas(cur, data_definition=dd)

    report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset()
    ])

    result = report.run(reference_data=ds_ref, current_data=ds_cur)
    output_path = "evidently_report.html"
    result.save_html(output_path)
    print("✅ Evidently report generated: evidently_report.html")

    ref_metrics = calc_metrics(ref[target], ref["prediction"])
    cur_metrics = calc_metrics(cur[target], cur["prediction"])

    # Define degraded metrics based on threshold (example: accuracy drop > 0.05)
    degraded = []
    for k in ref_metrics:
        if k in cur_metrics:
            # Example threshold: 5% drop for accuracy, precision, recall, f1
            if k in ["Accuracy", "Precision", "Recall", "F1_Score"]:
                if ref_metrics[k] - cur_metrics[k] > 0.1:
                    degraded.append(k)
    decision = "YES" if degraded else "NO"
    rationale = f"Threshold: 10% Degradation. Degraded metrics: {', '.join(degraded)}" if degraded else "All metrics within threshold. Threshold: 10% Degradation. "

    pd.DataFrame({
        "Retraining_Decision": [decision],
        "Rationale": [rationale]
    }).to_csv("Retrain.csv", index=False)

    
    with mlflow.start_run(run_name="Monitoring_Champion") as run:
        mlflow.log_artifact("evidently_report.html")
        # mlflow.log_artifact("metrics.json")
        mlflow.log_artifact("Retrain.csv")
        for k,v in cur_metrics.items():
            mlflow.log_metric(f"Current_{k}", v)
        for k,v in ref_metrics.items():
            mlflow.log_metric(f"Reference_{k}", v)
        mlflow.set_tag("Retrain_Decision", decision)
        mlflow.set_tag("Rationale", rationale)
        mlflow.set_tag("Model_Stage", "Production")
        mlflow.set_tag("Model_Role", "Champion")

    print("Monitoring complete. Report and metrics logged to MLflow.")

if __name__ == "__main__":
    main()
