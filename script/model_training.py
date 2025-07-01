# script/model_training.py

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings("ignore")

# ✅ Use root-level 'mlruns' directory
mlflow.set_tracking_uri("file:///C:/Users/mijuu/Documents/credit_risk_probability/mlruns")
mlflow.set_experiment("credit_risk_modeling")


def train_models(data_path="../data/final_model_dataset.csv"):
    """
    Load data, train models, evaluate, and track using MLflow.
    """

    # Step 1: Load the dataset
    df = pd.read_csv(data_path)

    # Step 2: Separate features and target
    X = df.drop(columns=["CustomerId", "is_high_risk"], errors="ignore")
    y = df["is_high_risk"]

    # Step 3: Split into train-test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 4: Define models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    # Step 5: Start MLflow experiment
    mlflow.set_experiment("CreditRiskModel")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            # Train model
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Log parameters and metrics
            mlflow.log_param("model_name", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            # Log model
            mlflow.sklearn.log_model(model, f"{name}_model")

            print(f"\n📊 {name} Metrics:")
            print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

            # Optional: Save model locally
            # import joblib
            # joblib.dump(model, f"../models/{name.lower()}_model.pkl")

def tune_random_forest(X_train, y_train, X_val, y_val):
    # Define the model
    rf = RandomForestClassifier(random_state=42)

    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }

    # Grid Search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='f1',
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    print("🔍 Starting hyperparameter tuning...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Log results in MLflow
    with mlflow.start_run(run_name="RandomForest_Tuned"):
        mlflow.sklearn.log_model(best_model, "rf_model_tuned")
        mlflow.log_params(grid_search.best_params_)

        preds = best_model.predict(X_val)
        report = classification_report(y_val, preds, output_dict=True)

        mlflow.log_metrics({
            "accuracy": report["accuracy"],
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1": report["1"]["f1-score"]
        })

        print("✅ Best Parameters Found:", grid_search.best_params_)
        print("📊 Classification Report:\n", classification_report(y_val, preds))

    return best_model
