"""Classification metrics with explicit abstention coverage."""


def classification_report(truth, predictions, labels):
    if not truth or len(truth) != len(predictions):
        raise ValueError("Require nonempty paired labels")
    if len(set(labels)) != len(labels) or not labels or "unknown" in labels:
        raise ValueError("Labels must be unique, nonempty, and exclude unknown")
    if any(value not in labels for value in truth):
        raise ValueError("Ground truth contains a label outside the label set")
    if any(value not in labels and value != "unknown" for value in predictions):
        raise ValueError("Predictions contain a label outside the label set")
    columns = list(labels) + ["unknown"]
    matrix = [[sum(a == actual and b == predicted for a, b in zip(truth, predictions))
               for predicted in columns] for actual in labels]
    per_class = {}
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(truth, predictions))
        fp = sum(a != label and b == label for a, b in zip(truth, predictions))
        fn = sum(a == label and b != label for a, b in zip(truth, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = {"precision": precision, "recall": recall,
                            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                            "support": truth.count(label)}
    return {"samples": len(truth), "labels": list(labels), "prediction_columns": columns,
            "accuracy": sum(a == b for a, b in zip(truth, predictions)) / len(truth),
            "macro_f1": sum(row["f1"] for row in per_class.values()) / len(labels),
            "coverage": sum(value != "unknown" for value in predictions) / len(truth),
            "per_class": per_class, "confusion_matrix": matrix,
            "unknown_policy": "Unknown predictions count as false negatives, never silently excluded."}
