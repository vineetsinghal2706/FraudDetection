import json
import os
import pandas as pd
import snowflake.connector
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix
)
import joblib

# Load credentials from environment variables
account = os.getenv('SNOWFLAKE_ACCOUNT')
user = os.getenv('SNOWFLAKE_USER')
password = os.getenv('SNOWFLAKE_PASSWORD')
warehouse = os.getenv('SNOWFLAKE_WAREHOUSE')
database = os.getenv('SNOWFLAKE_DATABASE')  # should be 'CREDITCARD'
schema = os.getenv('SNOWFLAKE_SCHEMA')      # should be 'PUBLIC'

# Function to fetch data from original table
def fetch_data_from_snowflake():
    conn = snowflake.connector.connect(
        user=user,
        password=password,
        account=account,
        warehouse=warehouse,
        database=database,
        schema=schema
    )
    cur = conn.cursor()
    cur.execute("SELECT * FROM CREDITCARD.PUBLIC.CREDITCARD")
    df = cur.fetch_pandas_all()
    conn.close()
    return df


def main():
    # Step 1: Load data
    data = fetch_data_from_snowflake()
    print("✅ Data loaded from Snowflake. Shape:", data.shape)

    # Step 2: Split features and target
    X = data.drop(['CLASS'], axis=1)
    y = data['CLASS']
    print("\n🎯 Features shape:", X.shape)
    print("🎯 Target shape:", y.shape)

    # Step 3: Train-test split
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)
    print("✅ Data split into train and test sets.")

    # Step 4: Train model
    rfc = RandomForestClassifier()
    rfc.fit(xTrain, yTrain)
    print("✅ Random Forest model trained.")

    # Step 5: Evaluate model
    yPred = rfc.predict(xTest)
    metrics = {
        'Accuracy': accuracy_score(yTest, yPred),
        'Precision': precision_score(yTest, yPred),
        'Recall': recall_score(yTest, yPred),
        'F1 Score': f1_score(yTest, yPred),
        'Matthews Corrcoef': matthews_corrcoef(yTest, yPred)
    }

    print("\n📊 Model Evaluation Metrics:")
    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")

    # Confusion matrix
    print("\n📉 Confusion Matrix:")
    print(confusion_matrix(yTest, yPred))

    # Dump to JSON
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("✅ Metrics dumped to metrics.json")
    # Step 6: Save model
    model_path = "model.pkl"
    joblib.dump(rfc, model_path)
    print(f"\n✅ Model saved to: {model_path}")

    
    print("\n🏁 All steps completed successfully.")

if __name__ == "__main__":
    main()
#runagain
