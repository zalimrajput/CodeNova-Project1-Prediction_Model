# Adult Income Prediction Using Machine Learning

## 1. Project Overview

This project develops a machine learning classification system to predict whether an individual's annual income exceeds $50,000 using demographic, education, employment, and financial attributes from the UCI Adult Census Income dataset.

The project implements data preprocessing, feature engineering, baseline model comparison, hyperparameter tuning, evaluation visualization, and cost-sensitive classification threshold optimization using Out-of-Fold (OOF) predictions.

The primary goal is to build a reproducible classification pipeline and evaluate the trade-off between correctly identifying higher-income individuals and minimizing false-positive predictions.

## 2. Project Objectives

- Build a complete model training and evaluation codebase.
- Compare multiple machine learning algorithms against a baseline.
- Improve model selection using validation metrics and hyperparameter tuning.
- Generate ROC, precision-recall, model comparison, and threshold evaluation charts.
- Optimize the decision threshold based on an explicit false-negative/false-positive cost assumption.
- Avoid target leakage during feature engineering, preprocessing, model selection, and evaluation.
- Document model limitations, failure cases, assumptions, and potential fairness concerns.

## 3. Dataset

**Dataset:** UCI Adult Census Income Dataset

**Source:** https://archive.ics.uci.edu/dataset/2/adult

The dataset contains demographic and employment information used to predict an income category.

| Dataset split | Records |
|---|---:|
| Official training dataset | 32,561 |
| Official test dataset | 16,281 |

### Target variable

- `<=50K` → Class 0
- `>50K` → Class 1

The official training and test files are preserved. Trailing periods in the official test labels are removed to make target labels consistent.

### Class distribution

| Split | Income `<=50K` | Income `>50K` |
|---|---:|---:|
| Training | 24,720 | 7,841 |
| Test | 12,435 | 3,846 |

The target is imbalanced, with fewer observations in the `>50K` class. Consequently, accuracy alone is not sufficient for assessing model quality.

## 4. Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Google Colab

## 5. Methodology

### 5.1 Data Preprocessing

The preprocessing workflow includes:

- Loading the official training and test datasets.
- Standardizing income labels across both splits.
- Handling missing numerical values using median imputation.
- Handling missing categorical values using an `Unknown` category.
- Scaling numerical features using `StandardScaler`.
- Encoding categorical features using `OneHotEncoder(handle_unknown="ignore")`.
- Removing exact duplicate rows as part of data-quality analysis, while preserving the official train/test separation.

The reported duplicate counts were 24 in the training dataset and 5 in the test dataset.

### 5.2 Feature Engineering

Numerical features include age, final weight (`fnlwgt`), education level encoded as `education_num`, capital gain, capital loss, and hours worked per week.

Additional features:

- `capital_gain_log`
- `capital_loss_log`
- `has_capital_gain`
- `has_capital_loss`

The logarithmic features represent highly skewed financial variables, while the indicator features identify whether capital gains or losses are nonzero.

The categorical features include workclass, marital status, occupation, relationship, race, sex, and native country.

The original `education` field is excluded because it overlaps substantially with `education_num`.

### 5.3 Leakage Prevention

The official test set is not used for feature selection, hyperparameter tuning, threshold selection, or preprocessing fitting.

To maintain leakage-safe evaluation:

- Fit imputers, scalers, and encoders only on the relevant training data.
- Use a Scikit-learn `Pipeline` or equivalent fold-specific preprocessing so validation rows do not influence preprocessing statistics.
- Perform hyperparameter tuning using training folds and validation metrics.
- Generate OOF predictions so each training observation is predicted by a model that did not train on that observation.
- Select the final threshold using training OOF predictions, not test labels.
- Evaluate the final fitted model on the untouched official test set.

The target variable is used only as the supervised learning label and for evaluation. It is not used to construct input features.

### 5.4 Models Compared

Four candidate models were evaluated using a default threshold of 0.50:

1. Logistic Regression — baseline model.
2. Random Forest — nonlinear tree-based model.
3. XGBoost — gradient-boosted tree model.
4. Tuned XGBoost — XGBoost with optimized hyperparameters.

Hyperparameter tuning used RandomizedSearchCV with 20 parameter candidates, three-fold cross-validation, and average precision (PR-AUC) as the optimization metric.

### 5.5 Tuned XGBoost Configuration

The selected hyperparameters were:

| Parameter | Value |
|---|---:|
| `n_estimators` | 600 |
| `max_depth` | 5 |
| `learning_rate` | 0.03 |
| `subsample` | 1.0 |
| `colsample_bytree` | 0.8 |
| `min_child_weight` | 1 |
| `gamma` | 0.1 |
| `reg_lambda` | 0.5 |

## 6. Baseline Comparison

All four candidate models were initially evaluated on the held-out validation split at the default threshold of 0.50.

| Metric | Logistic Regression | Random Forest | XGBoost | Tuned XGBoost |
|---|---:|---:|---:|---:|
| Accuracy | 85.95% | 87.16% | 87.76% | 87.72% |
| Precision | 74.79% | 80.25% | 78.79% | 79.14% |
| Recall | 62.82% | 61.93% | 67.28% | 66.52% |
| F1-score | 68.28% | 69.91% | 72.58% | 72.28% |
| ROC-AUC | 0.9137 | 0.9244 | 0.9320 | 0.9319 |
| PR-AUC | 0.7887 | 0.8237 | 0.8374 | **0.8394** |

### Model selection

Tuned XGBoost was selected because it achieved the highest validation PR-AUC of **0.8394**.

Compared with Logistic Regression, tuned XGBoost improved validation PR-AUC by 0.0507, or approximately 6.4% relative to the baseline PR-AUC. It also improved validation ROC-AUC from 0.9137 to 0.9319.

The default XGBoost model had slightly better recall, F1-score, and ROC-AUC than tuned XGBoost. The tuned model was selected according to the predefined primary metric, PR-AUC, rather than because it outperformed every other model on every metric.

## 7. OOF-Based Threshold Optimization

After selecting tuned XGBoost, five-fold cross-validation was used to generate Out-of-Fold (OOF) probabilities for the official training dataset.

Each training observation receives a probability from a model that did not train on that observation.

The threshold was selected by minimizing the assumed classification cost:

`Cost = 5 × False Negatives + False Positives`

The selected threshold was **0.16** under this cost function.

The cost ratio is an experimental assumption, not an intrinsic property of the dataset. A different relative cost for false negatives and false positives may result in a different optimal threshold.

## 8. Final Test Results

The selected tuned XGBoost model was refitted on the full official training dataset and evaluated on the official test dataset.

The following results compare the default threshold of 0.50 with the OOF-selected threshold of 0.16.

| Metric | Threshold 0.50 | Threshold 0.16 |
|---|---:|---:|
| Accuracy | 87.54% | 79.53% |
| Precision | 78.46% | 53.90% |
| Recall | 65.16% | **92.07%** |
| F1-score | 71.19% | 68.00% |
| F2-score | 67.45% | **80.65%** |
| True positives | 2,506 | 3,541 |
| True negatives | 11,747 | 9,407 |
| False positives | 688 | 3,028 |
| False negatives | 1,340 | **305** |
| Weighted cost | 7,388 | **4,553** |

### Interpretation

At threshold 0.16, recall increased from 65.16% to 92.07%, while false negatives decreased from 1,340 to 305.

Under the assumed 5:1 cost ratio, the test-set weighted cost decreased from 7,388 to 4,553, a retrospective reduction of approximately 38.4%.

This improvement comes with a trade-off: false positives increased from 688 to 3,028, precision declined, and overall accuracy fell.

The threshold of 0.16 is therefore appropriate only when the priority is to identify more members of the `>50K` class and the increased false-positive rate is acceptable. The test results illustrate the trade-off; the test set was not used to choose the threshold.

ROC-AUC and PR-AUC are threshold-independent ranking metrics and should be reported separately from threshold-dependent metrics.

## 9. Evaluation Charts

The project should include charts generated from the actual evaluation code and results:

| Chart | Purpose |
|---|---|
| Model comparison chart | Compare candidate models using validation PR-AUC, ROC-AUC, F1, or recall. |
| ROC curve | Visualize the trade-off between true-positive and false-positive rates. |
| Precision-recall curve | Evaluate performance on the imbalanced target class. |
| Threshold comparison chart | Show how precision, recall, F1, F2, and weighted cost change with the decision threshold. |
| Feature importance chart | Show the most influential features for the fitted tree-based model. |


These files must be generated and committed before claiming that the corresponding charts are included.

## 10. How to Run

### Obtain the dataset

Download the Adult dataset from the official UCI repository:

https://archive.ics.uci.edu/dataset/2/adult

Place `adult.data`, `adult.test`, and any required metadata files in the locations expected by the notebook or training scripts.

### Run the project

If using the Colab notebook, open `adult_income_prediction.ipynb` in Google Colab, upload the required dataset files, and run the cells in order.

If using the modular source code, follow the commands documented for the actual training and evaluation entry points.

## 11. Known Limitations and Failure Cases

### Class imbalance

The `>50K` class is the minority class. Accuracy can therefore conceal poor minority-class recall. PR-AUC, recall, precision, F1/F2, and the confusion matrix provide additional insight.

### Threshold trade-offs

A lower threshold increases positive predictions. It may identify more true positives but can also increase false positives substantially. The chosen threshold depends on the application and its error costs.

### Dataset age and representativeness

The Adult dataset is based on historical census data. Relationships learned from it may not represent contemporary income distributions or other populations.

### Feature limitations

The available attributes cannot capture every factor that influences income. Missing, coarse, or imperfect attributes can lead to incorrect predictions.

### Fairness concerns

Features such as sex and race can introduce or reflect historical inequalities. Model performance must be assessed across relevant groups before any real-world use. Removing sensitive features alone does not guarantee fairness.

### Model interpretability

XGBoost can learn complex nonlinear relationships, but individual predictions are less straightforward to explain than those of a simple linear model. Feature importance is not proof of causation.

### Cost assumptions

The 5:1 false-negative/false-positive cost ratio is an assumed objective. Different costs can change the selected threshold and the preferred operating point.

### Generalization

The reported scores measure performance on this dataset and split. They do not guarantee performance on future data or other populations. External validation and ongoing monitoring would be required for practical deployment.

## 12. Assumptions

- The task is binary classification of income category.
- The official UCI train/test split is preserved.
- The primary selection metric is validation PR-AUC.
- All candidate models use threshold 0.50 during the initial baseline comparison.
- OOF probabilities are used to select a threshold for the selected tuned XGBoost model.
- The threshold objective assumes five times the cost for a false negative compared with a false positive.
- The final official test set is reserved for final evaluation.

## 13. Deliverables

- Model training and evaluation codebase.
- Data preprocessing and feature engineering workflow.
- Baseline comparison against Logistic Regression.
- Random Forest, XGBoost, and tuned XGBoost evaluation.
- OOF-based decision threshold optimization.
- Evaluation charts and benchmark results.
- README documentation of methodology, metrics, assumptions, and limitations.
- Leakage-aware train/validation/test methodology.

## 14. Conclusion

This project demonstrates an end-to-end Adult Income classification workflow using Python and machine learning. It compares baseline and tree-based models, selects tuned XGBoost using validation PR-AUC, and applies OOF-based threshold optimization to investigate the trade-off between false negatives and false positives.

The final test evaluation demonstrates that threshold selection can materially change the classifier's behavior. The project also documents the limits of the dataset, the assumptions behind the cost function, and the importance of leakage prevention and fairness assessment.

## Author

Add your name and GitHub profile here.
