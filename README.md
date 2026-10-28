# Loan Eligibility Prediction

This project follows the supplied Dream Housing Finance assignment: predict whether an applicant is eligible for a home loan using the provided applicant details. The training data has 614 labeled applicants; the test data has 367 unlabeled applicants.

## What the project does

1. Reads the provided training and test CSV files from `data/`.
2. Uses `Loan_Status` (`Y`/`N`) as the target and excludes `Loan_ID` from the predictors.
3. Fills missing numeric values with medians and missing categorical values with the most common category; one-hot encodes categories.
4. Compares logistic regression and random forest on a stratified 20% validation split. The model with the higher ROC-AUC is selected.
5. Retrains that model on all labeled rows and writes 0/1 predictions matching the sample submission format.

The supplied attribute notes describe EMI, but neither CSV includes an EMI field. This implementation uses only fields actually present in the data. `Loan_ID` is preserved in the submission but is not used to make predictions.

## Project layout

```text
data/
  loan_train.csv
  loan_test.csv
  sample_submission.csv
  problem_statement.txt
  attribute_description.txt
loan_prediction_walkthrough.ipynb
src/
  train.py  # optional command-line version
requirements.txt
```

## Build it yourself in Jupyter

The guided notebook is `loan_prediction_walkthrough.ipynb`. It walks through loading and exploring the data, handling missing values, training and comparing two models, evaluating errors, and writing a sample submission. Run each cell in order and fill in the reflection prompts in Markdown cells.

From this project folder, create an environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter notebook
```

In the browser that opens, select `loan_prediction_walkthrough.ipynb`, then use **Run All** or run cells one by one. The notebook saves `outputs/submission.csv`. The optional command-line version is `python src/train.py`.

## How to explain the workflow

- **Problem type:** supervised binary classification.
- **Target:** `Loan_Status`; `Y` means eligible and `N` means not eligible.
- **Inputs:** gender, marital status, dependents, education, employment, applicant and co-applicant income, loan amount, loan term, credit history, and property area.
- **Evaluation:** ROC-AUC compares ranking quality across thresholds; the printed classification report also shows precision, recall, and F1 for each class.
- **Limitations:** the dataset is small and imbalanced (422 approvals and 192 rejections). The validation score is an estimate, and a model prediction is not a real lending decision. Review fairness, calibration, and applicable lending rules before any real use.

## Suggested next steps for a class presentation

Show the missing-value counts, the class balance, the validation metrics, and the model's most influential features. Discuss whether applicants should be assessed using applicant plus co-applicant income, and why credit history and loan terms may matter. Do not claim that validation results guarantee performance on future applicants.
