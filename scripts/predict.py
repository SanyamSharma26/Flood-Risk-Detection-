import pandas as pd
import joblib
import argparse
import sys

def main(input_csv, output_csv):
    # ----------------------------
    # 1. Load models
    # ----------------------------
    try:
        imputer = joblib.load("models/flood_imputer.pkl")
        scaler = joblib.load("models/flood_scaler.pkl")
        model = joblib.load("models/flood_risk_model.pkl")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        sys.exit(1)

    # ----------------------------
    # 2. Load input features
    # ----------------------------
    try:
        X_new = pd.read_csv(input_csv)
        print(f"✅ Loaded input data: {X_new.shape}")
    except Exception as e:
        print(f"❌ Error reading input CSV: {e}")
        sys.exit(1)

    # ----------------------------
    # 3. Preprocess
    # ----------------------------
    try:
        X_imputed = imputer.transform(X_new)
        X_scaled = scaler.transform(X_imputed)
    except Exception as e:
        print(f"❌ Error preprocessing data: {e}")
        sys.exit(1)

    # ----------------------------
    # 4. Predict flood risk
    # ----------------------------
    try:
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]  # probability of flood
    except Exception as e:
        print(f"❌ Error predicting: {e}")
        sys.exit(1)

    # ----------------------------
    # 5. Risk level assignment
    # ----------------------------
    risk_levels = pd.qcut(
        y_prob, q=3, labels=["low", "medium", "high"]
    )  # adaptive bins

    # ----------------------------
    # 6. Save results
    # ----------------------------
    results = X_new.copy()
    results["flooded"] = y_pred
    results["flood_probability"] = y_prob
    results["risk_level"] = risk_levels

    results.to_csv(output_csv, index=False)
    print(f"✅ Predictions saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flood Risk Prediction Script")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path to save output CSV")

    args = parser.parse_args()
    main(args.input, args.output)
