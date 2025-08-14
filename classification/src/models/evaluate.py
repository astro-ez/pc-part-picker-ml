import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, matthews_corrcoef, balanced_accuracy_score,
    roc_curve, precision_recall_curve
)
import seaborn as sns
import yaml

def evaluate_model(model, X_test, y_test, metrics_path="models/metrics.json", reports_dir="reports"):
    """Evaluate classification model and save metrics + plots."""

    # Load parameters
    params = yaml.safe_load(open('params/evaluate.yaml', 'r'))
    # Predict class labels and probabilities
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)  # probability for positive class
    else:
        y_proba = y_pred  # fallback for models without predict_proba

    # Reshape the y_proba 
    # y_proba = np.array(y_proba).reshape(-1, 1) if y_proba.ndim == 1 else y_proba

    # Reshape the y_test
    # y_test = np.array(y_test).reshape(-1, 1) if y_test.ndim == 1 else y_test

    
    # Calculate metrics
    # Assuming you have a mapping from class index to label:
    classes = np.unique(y_test)  # or use your id2label mapping

    def to_serializable(val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        elif isinstance(val, (np.generic,)):
            return val.item()
        else:
            return val

    # Example for per-class metrics
    precision_per_class = dict(zip(classes, precision_score(y_test, y_pred, average=None, zero_division=0)))
    recall_per_class = dict(zip(classes, recall_score(y_test, y_pred, average=None, zero_division=0)))
    f1_per_class = dict(zip(classes, f1_score(y_test, y_pred, average=None, zero_division=0)))

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_per_class,
        "recall": recall_per_class,
        "f1_score": f1_per_class,
        "roc_auc": roc_auc_score(y_test, y_proba) if len(classes) == 2 else None,
        "pr_auc": average_precision_score(y_test, y_proba, average=params['average']),
        "log_loss": log_loss(y_test, y_proba),
        "specificity": None,
        "mcc": matthews_corrcoef(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred)
    }

    # Convert everything to JSON-friendly format
    metrics_serializable = {k: to_serializable(v) for k, v in metrics.items()}

    # Calculate specificity from confusion matrix
    # tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # metrics["specificity"] = tn / (tn + fp)

    # Save metrics.json
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics_serializable, f, indent=4)

    # # Create reports directory
    # os.makedirs(reports_dir, exist_ok=True)

    # # Confusion Matrix Plot
    # plt.figure(figsize=(6, 5))
    # sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    # plt.xlabel("Predicted"); plt.ylabel("True")
    # plt.title("Confusion Matrix")
    # plt.savefig(f"{reports_dir}/confusion_matrix.png")
    # plt.close()

    # # ROC Curve
    # if metrics["roc_auc"] is not None:
    #     fpr, tpr, _ = roc_curve(y_test, y_proba)
    #     plt.figure(figsize=(6, 5))
    #     plt.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.2f}")
    #     plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    #     plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    #     plt.title("ROC Curve")
    #     plt.legend()
    #     plt.savefig(f"{reports_dir}/roc_curve.png")
    #     plt.close()

    # # Precision-Recall Curve
    # prec, rec, _ = precision_recall_curve(y_test, y_proba)
    # plt.figure(figsize=(6, 5))
    # plt.plot(rec, prec, label=f"PR AUC={metrics['pr_auc']:.2f}")
    # plt.xlabel("Recall"); plt.ylabel("Precision")
    # plt.title("Precision-Recall Curve")
    # plt.legend()
    # plt.savefig(f"{reports_dir}/pr_curve.png")
    # plt.close()

    print(f"Metrics saved to {metrics_path}")
    print(f"Plots saved in {reports_dir}/")
