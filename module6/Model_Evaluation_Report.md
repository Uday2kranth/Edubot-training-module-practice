# Model Evaluation Report: Binary and Multi-Class Classification

## Objective

The goal of this report is to evaluate and compare the performance of three classification models:

- Logistic Regression
- k-Nearest Neighbors (k-NN)
- Decision Tree

Each model is tested on both:

- Binary classification tasks
- Multi-class classification tasks

The evaluation is based on standard metrics and visualizations. The report provides comparative analysis and concludes with model recommendations.

## Binary Classification

### Evaluation Metrics (Extracted)

- **Logistic Regression**
  - Accuracy: 0.91
  - Precision: 0.90
  - Recall: 0.92
  - F1-Score: 0.91

- **k-Nearest Neighbors (k-NN)**
  - Accuracy: 0.87
  - Precision: 0.86
  - Recall: 0.88
  - F1-Score: 0.87

- **Decision Tree**
  - Accuracy: 0.86
  - Precision: 0.84
  - Recall: 0.88
  - F1-Score: 0.86

### Visual Analysis

- Confusion matrices were generated to show prediction accuracy across the two classes.
- ROC Curves:
  - Logistic Regression showed the highest AUC value, close to 1.0.
  - k-NN and Decision Tree had lower AUC scores indicating weaker discrimination.

### Comparative Analysis

- Logistic Regression outperformed other models across all evaluation metrics.
- k-NN had strong recall but lower precision compared to Logistic Regression.
- Decision Tree provided fair results but was slightly less effective than the other models.

### Recommendation for Binary Classification

- **Logistic Regression** is recommended due to its consistent and superior performance across all metrics and its simplicity in training and interpretation.

## Multi-Class Classification

### Evaluation Metrics (Macro-Averaged)

- **Logistic Regression**
  - Accuracy: 0.83
  - Precision: 0.81
  - Recall: 0.82
  - F1-Score: 0.81

- **k-Nearest Neighbors (k-NN)**
  - Accuracy: 0.77
  - Precision: 0.75
  - Recall: 0.74
  - F1-Score: 0.74

- **Decision Tree**
  - Accuracy: 0.79
  - Precision: 0.76
  - Recall: 0.78
  - F1-Score: 0.77

### Visual Analysis

- Confusion matrices revealed misclassifications in closely related classes.
- ROC Curves (One-vs-Rest):
  - Logistic Regression again showed better class separability.
  - k-NN struggled with overlapping class boundaries.
  - Decision Tree performed better than k-NN but worse than Logistic Regression.

### Comparative Analysis

- Logistic Regression showed the most balanced performance across all metrics.
- Decision Tree was effective, especially in capturing non-linear relationships.
- k-NN's performance suffered due to sensitivity to class imbalance and input dimensionality.

### Recommendation for Multi-Class Classification

- **Primary Recommendation**: Logistic Regression for general multi-class tasks.
- **Secondary Recommendation**: Decision Tree for cases requiring rule-based interpretation or handling non-linear patterns.

## Overall Conclusion

### Final Recommendations

- **Binary Classification**
  - Best Model: Logistic Regression

- **Multi-Class Classification**
  - Best Model: Logistic Regression
  - Alternative (if interpretability is a concern): Decision Tree

### Summary

- Logistic Regression performed best overall in both binary and multi-class scenarios.
- Decision Tree is a viable backup for multi-class classification, especially when model interpretability or non-linear separation is needed.
- k-NN, although simple and intuitive, did not outperform the other two models and may not be ideal for high-dimensional or imbalanced datasets.



## Notes

- All models were trained and evaluated on the same dataset split.
- Evaluation included both quantitative metrics (accuracy, precision, recall, F1-score) and qualitative insights (confusion matrices, ROC curves).
- Multi-class evaluation used macro-averaging to ensure fair performance comparison across all classes.
