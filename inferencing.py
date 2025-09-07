import os
import pandas as pd
import snowflake.connector
import joblib  # ✅ Use joblib or pickle to load the local .pkl model
import sys
import io

# Fix Windows stdout encoding issue (for Windows terminals)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Snowflake credentials
SNOWFLAKE_ACCOUNT = os.getenv('SNOWFLAKE_ACCOUNT')
SNOWFLAKE_USER = os.getenv('SNOWFLAKE_USER')
SNOWFLAKE_PASSWORD = os.getenv('SNOWFLAKE_PASSWORD')
SNOWFLAKE_WAREHOUSE = os.getenv('SNOWFLAKE_WAREHOUSE')
SNOWFLAKE_DATABASE = os.getenv('SNOWFLAKE_DATABASE')
SNOWFLAKE_SCHEMA = os.getenv('SNOWFLAKE_SCHEMA')

# Table names
BATCH_INPUT_TABLE = f"{SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.CREDITCARD_BATCH_INPUTS"
BATCH_PREDICTIONS_TABLE = f"{SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.BATCH_PREDICTIONS"

def get_snowflake_connection():
    return snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA
    )

def fetch_batch_data():
    print(f"📥 Fetching batch data from Snowflake table: {BATCH_INPUT_TABLE}")
    with get_snowflake_connection() as conn:
        df = pd.read_sql(f"SELECT * FROM {BATCH_INPUT_TABLE}", conn)
        print(f"✅ Fetched {df.shape[0]} rows and {df.shape[1]} columns.")
        return df

def get_champion_model():
    model_path = "champion_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Could not find champion model at '{model_path}'")
    
    print(f"🎯 Loading champion model from local file: {model_path}")
    model = joblib.load(model_path)
    return model

def generate_predictions(df, model):
    # Ensure ID column exists
    if 'ID' not in df.columns:
        df.insert(0, 'ID', range(1, len(df) + 1))

    features = df.drop(columns=['ID'] + (['CLASS'] if 'CLASS' in df.columns else []))

    print(f"🔍 Generating predictions for {features.shape[0]} records...")

    preds = model.predict(features)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[:, 1]
    else:
        probs = [None] * len(preds)

    result_df = df.copy()
    result_df['PREDICTION'] = preds
    result_df['PREDICTION_PROB'] = probs

    return result_df

def save_predictions_to_snowflake(df):
    print(f"🧹 Truncating and inserting predictions into {BATCH_PREDICTIONS_TABLE}...")
    with get_snowflake_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(f"TRUNCATE TABLE {BATCH_PREDICTIONS_TABLE}")
            conn.commit()

            cols = list(df.columns)
            placeholders = ', '.join(['%s'] * len(cols))
            insert_query = f"INSERT INTO {BATCH_PREDICTIONS_TABLE} ({', '.join(cols)}) VALUES ({placeholders})"
            data = [tuple(row) for row in df.to_numpy()]
            cursor.executemany(insert_query, data)
            conn.commit()

            print("✅ Predictions successfully inserted into Snowflake.")
        finally:
            cursor.close()

def main():
    print("🚀 Starting batch inference...")
    batch_df = fetch_batch_data()
    model = get_champion_model()
    predictions_df = generate_predictions(batch_df, model)
    save_predictions_to_snowflake(predictions_df)
    print("🏁 Batch inference pipeline completed.")

if __name__ == "__main__":
    main()
