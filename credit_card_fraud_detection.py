"""
CODSOFT - TASK 2: CREDIT CARD FRAUD DETECTION
==============================================
Dataset: Credit Card Transactions Fraud Detection Dataset (Kaggle)
Files used: fraudTrain.csv, fraudTest.csv
Target column: is_fraud (0 = Legitimate, 1 = Fraud)

Approach:
 1. Load fraudTrain.csv and fraudTest.csv (dataset already comes pre-split)
 2. Engineer useful features:
      - customer age (from date of birth)
      - transaction hour / day-of-week / month
      - distance between customer location and merchant location
 3. Drop identifying / high-cardinality columns not useful for modeling
    (names, card number, street, transaction id, raw timestamps)
 4. Encode categorical features (category, gender, state) and scale amount
 5. Balance the TRAINING data only via undersampling (test set stays realistic)
 6. Train 3 classifiers: Logistic Regression, Decision Tree, Random Forest
 7. Evaluate with Precision, Recall, F1-score, ROC-AUC (accuracy is misleading
    here because fraud is a small minority of transactions)
 8. Save best model + charts
"""

import os
import pickle
import numpy as np
import pandas as pd

# Always look for files in the same folder as this script,
# regardless of where the terminal's working directory happens to be.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
TRAIN_PATH = "fraudTrain.csv"
TEST_PATH = "fraudTest.csv"
RANDOM_STATE = 42


# -------------------------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()

    # --- Age from date of birth ---
    df["dob"] = pd.to_datetime(df["dob"])
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365

    # --- Time-based features ---
    df["trans_hour"] = df["trans_date_trans_time"].dt.hour
    df["trans_day"] = df["trans_date_trans_time"].dt.dayofweek  # 0=Mon
    df["trans_month"] = df["trans_date_trans_time"].dt.month

    # --- Distance between customer and merchant (approx, in degrees->km) ---
    df["distance"] = np.sqrt(
        (df["lat"] - df["merch_lat"]) ** 2 + (df["long"] - df["merch_long"]) ** 2
    ) * 111  # rough conversion: 1 degree ~ 111 km

    # --- Drop columns that don't help / leak identity ---
    drop_cols = [
        "Unnamed: 0", "trans_date_trans_time", "cc_num", "first", "last",
        "street", "trans_num", "unix_time", "dob", "zip", "merchant", "city", "job",
        "lat", "long", "merch_lat", "merch_long",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


def encode_categoricals(train_df, test_df, cat_cols):
    """Fit LabelEncoders on combined train+test categories to avoid unseen-label errors."""
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        encoders[col] = le
    return train_df, test_df, encoders


def main():
    # -----------------------------------------------------------
    # STEP 1: LOAD DATA (already pre-split by the dataset provider)
    # -----------------------------------------------------------
    print("Loading dataset...")
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    print(f"Train: {len(train_raw)} transactions | Test: {len(test_raw)} transactions")

    fraud_count = train_raw["is_fraud"].sum()
    fraud_pct = (fraud_count / len(train_raw)) * 100
    print(f"Fraud in training set: {fraud_count} ({fraud_pct:.3f}%)")

    plt.figure(figsize=(5, 4))
    train_raw["is_fraud"].value_counts().plot(kind="bar", color=["#4C72B0", "#C44E52"])
    plt.xticks([0, 1], ["Legitimate", "Fraud"], rotation=0)
    plt.ylabel("Count")
    plt.title("Class Distribution (Highly Imbalanced)")
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------
    # STEP 2: FEATURE ENGINEERING
    # -----------------------------------------------------------
    print("\nEngineering features (age, time, distance)...")
    train_df = engineer_features(train_raw)
    test_df = engineer_features(test_raw)

    # -----------------------------------------------------------
    # STEP 3: ENCODE CATEGORICALS
    # -----------------------------------------------------------
    print("Encoding categorical features...")
    cat_cols = ["category", "gender", "state"]
    train_df, test_df, encoders = encode_categoricals(train_df, test_df, cat_cols)

    # -----------------------------------------------------------
    # STEP 4: SCALE 'amt'
    # -----------------------------------------------------------
    scaler = StandardScaler()
    train_df["amt"] = scaler.fit_transform(train_df[["amt"]])
    test_df["amt"] = scaler.transform(test_df[["amt"]])

    X_train, y_train = train_df.drop("is_fraud", axis=1), train_df["is_fraud"]
    X_test, y_test = test_df.drop("is_fraud", axis=1), test_df["is_fraud"]

    # -----------------------------------------------------------
    # STEP 5: BALANCE TRAINING DATA (undersample majority class only)
    # -----------------------------------------------------------
    print("\nBalancing training data using undersampling...")
    train_bal = pd.concat([X_train, y_train], axis=1)
    fraud_train = train_bal[train_bal["is_fraud"] == 1]
    legit_train = train_bal[train_bal["is_fraud"] == 0].sample(
        n=len(fraud_train) * 3, random_state=RANDOM_STATE
    )
    balanced = pd.concat([fraud_train, legit_train]).sample(frac=1, random_state=RANDOM_STATE)
    X_train_bal = balanced.drop("is_fraud", axis=1)
    y_train_bal = balanced["is_fraud"]
    print(f"Balanced training set size: {len(X_train_bal)} "
          f"(Fraud: {y_train_bal.sum()}, Legit: {len(y_train_bal) - y_train_bal.sum()})")

    # -----------------------------------------------------------
    # STEP 6: TRAIN MULTIPLE MODELS
    # -----------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    results = {}
    trained_models = {}
    roc_data = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_bal, y_train_bal)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}
        trained_models[name] = model
        roc_data[name] = roc_curve(y_test, probs)

        print(f"{name} -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | "
              f"Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

    # -----------------------------------------------------------
    # STEP 7: PICK BEST MODEL (by F1-score)
    # -----------------------------------------------------------
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = trained_models[best_name]
    print(f"\nBest model (by F1-score): {best_name}")

    best_preds = best_model.predict(X_test)
    print("\nClassification Report (best model):")
    print(classification_report(y_test, best_preds, target_names=["Legitimate", "Fraud"]))

    # -----------------------------------------------------------
    # STEP 8: SAVE COMPARISON CHART
    # -----------------------------------------------------------
    metrics_df = pd.DataFrame(results).T
    metrics_df[["precision", "recall", "f1", "roc_auc"]].plot(
        kind="bar", figsize=(9, 5), colormap="Set2"
    )
    plt.title("Model Comparison (Precision / Recall / F1 / ROC-AUC)")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------
    # STEP 9: CONFUSION MATRIX FOR BEST MODEL
    # -----------------------------------------------------------
    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legitimate", "Fraud"], yticklabels=["Legitimate", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------
    # STEP 10: ROC CURVES FOR ALL MODELS
    # -----------------------------------------------------------
    plt.figure(figsize=(6, 5))
    for name, (fpr, tpr, _) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------
    # STEP 11: SAVE MODEL + SCALER
    # -----------------------------------------------------------
    with open("fraud_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("\nSaved fraud_model.pkl and scaler.pkl")

    return best_model, results


if __name__ == "__main__":
    best_model, results = main()
    print("\nDone. Charts saved: class_distribution.png, model_comparison.png, "
          "confusion_matrix.png, roc_curves.png")
