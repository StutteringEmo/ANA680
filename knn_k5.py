# ANA680 – Breast Cancer Wisconsin (Original)
# K-Nearest Neighbors (k=5) with scaling, 25% stratified test split

from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def main():
    # ---------- Fetch dataset ----------
    bc = fetch_ucirepo(id=15)

    # DataFrames
    X = bc.data.features
    y = bc.data.targets

    # (Verification) variable information
    print(bc.variables)

    # ---------- Clean ----------
    df = pd.concat([X, y], axis=1)
    df = df.replace("?", np.nan)
    for col in df.columns:
        if col != "Class":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(df.median(numeric_only=True))
    df["Class"] = df["Class"].map({2: 0, 4: 1})
    for cand in ["id", "ID", "Id", "Sample code number"]:
        if cand in df.columns:
            df = df.drop(columns=[cand])
            break

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # ---------- Split ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # ---------- Model ----------
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ])

    # ---------- Train & Evaluate ----------
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\nModel: KNN (k=5)")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

if __name__ == "__main__":
    main()