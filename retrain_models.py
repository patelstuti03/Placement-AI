"""
Retrain and re-save all model files with the scikit-learn version installed on
THIS machine.

Why this exists: the .pkl files shipped in models/ were created with an older
scikit-learn. Loading them with a newer version fails with errors such as
"No module named '_loss'". Running this script once fixes that permanently.

Usage:
    python retrain_models.py
"""
import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from app import (  # reuse the exact same definitions the app uses
    FEATURES, MODEL_FILES, MODEL_DIR, DATA_DIR, build_estimators,
)


def main():
    int_path = os.path.join(DATA_DIR, "cleaned_data_int0.csv")
    if not os.path.exists(int_path):
        raise SystemExit(f"Dataset not found: {int_path}")

    df = pd.read_csv(int_path, encoding="latin1")
    X = df[FEATURES].fillna(0)
    y = df["placedornot"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler().fit(X)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, estimator in build_estimators().items():
        estimator.fit(X_tr_s, y_tr)
        acc = accuracy_score(y_te, estimator.predict(X_te_s)) * 100
        out = os.path.join(MODEL_DIR, MODEL_FILES[name])
        joblib.dump(estimator, out)
        print(f"{name:<20} accuracy {acc:6.2f}%  ->  {out}")

    print("\nAll models retrained and saved.")


if __name__ == "__main__":
    main()
