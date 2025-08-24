import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

def evaluate_model(model, X_test, y_test, class_names=None):
    print("Evaluating model...")

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy & F1
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Classification report (converted to DataFrame)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_names)

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix Heatmap")
    plt.show()

    # Summary metrics
    summary = pd.DataFrame({
        "Metric": ["Accuracy", "F1 Score"],
        "Score": [acc, f1]
    })

    return report_df, summary

class_names = ["BFA", "BOTNET", "DDOS", "DOS", "NORMAL", "PROBE", "U2R", "WEB_ATTACK"]

report_df, summary = evaluate_model(model, X_test, y_test, class_names=class_names)

print("\nClassification Report:")
display(report_df)

print("\nSummary Table:")
display(summary)
