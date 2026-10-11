# Bank Marketing Classification: Predicting Term Deposit Subscriptions

Classification lab on the **UCI Bank Marketing** dataset. The goal is to predict whether a client will subscribe to a term deposit after a phone marketing campaign, so a Portuguese bank can decide **which clients to call first**.

I compare **Logistic Regression** and **K-Nearest Neighbors (KNN)**, handle a heavily imbalanced target with **SMOTENC**, avoid **data leakage**, and finish with a business recommendation.

---

## Project Overview

| | |
|---|---|
| **Dataset** | [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), the `bank.csv` 10% sample |
| **Rows / Columns** | 4,521 clients / 17 columns |
| **Target** | `y`: `yes` = subscribed, `no` = did not subscribe |
| **Class balance** | 88.5% `no` / 11.5% `yes` (imbalanced) |
| **Models** | Logistic Regression, KNN (k = 15) |
| **Main metric** | F1 score (positive class = `yes`) |

---

## Workflow

1. **Load and explore** the data (`;` separator, shape, dtypes, missing values, summary stats, target distribution).
2. **Define the business problem**: false positives are wasted calls (low cost); false negatives are lost customers (higher cost). F1 is chosen because both errors matter, and Accuracy is misleading on imbalanced data.
3. **Remove data leakage**: `duration` (call length) is only known *after* the call, so it is dropped.
4. **Encode binary columns** (`default`, `housing`, `loan`) to 0/1.
5. **Stratified 80/20 train/test split** (`random_state=42`).
6. **Preprocess**:
   - `StandardScaler` fitted on the training set only
   - **SMOTENC** applied to the training set only, before one-hot encoding (minority class raised to 70% of the majority class)
   - `OneHotEncoder` (`drop='first'`) fitted on the balanced training set
7. **Train and evaluate** Logistic Regression and KNN on the untouched test set.
8. **Interpret** the Logistic Regression coefficients as odds ratios.
9. **Compare** the models and make a recommendation.

---

## Results (Test Set)

| Model | Accuracy | Balanced Acc. | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.747 | 0.631 | 0.222 | 0.481 | **0.304** | 0.688 |
| KNN (k = 15) | 0.682 | 0.599 | 0.178 | 0.490 | 0.262 | 0.645 |

Both models are modest, which is expected: the strongest predictor (`duration`) was removed on purpose to avoid leakage and keep the model usable before a call is made.

### KNN: choosing k
Small values of `k` (3 to 5) memorize the training data (train F1 around 0.9 versus test F1 around 0.25). Test F1 plateaus from about k = 7, while the train/test gap keeps shrinking, so **k = 15** was chosen as a balance between performance and generalization.

### Logistic Regression: key drivers (odds ratios)

| Higher odds of subscribing | Odds ratio | Lower odds of subscribing | Odds ratio |
|---|---|---|---|
| `poutcome_success` | 9.94 | `loan` | 0.15 |
| `month_oct` | 4.42 | `contact_unknown` | 0.24 |
| `education_secondary` | 2.83 | `education_unknown` | 0.31 |
| `job_retired` | 2.80 | `job_unemployed` | 0.33 |

Reference categories are the first alphabetical level of each feature, and numerical features were standardized (one unit is about one standard deviation). A previous successful campaign is by far the strongest signal; having a personal loan is one of the strongest negative signals.

---

## Recommendation

I recommend **Logistic Regression** to the marketing team:

- Its F1 is higher than KNN's, and it also has better Precision, Accuracy and ROC AUC.
- It outputs a **probability per client**, which can be used to rank the call list.
- The **decision threshold can be tuned** to call-center capacity and the value of a subscription.
- Coefficients and odds ratios explain **why** a client is more or less likely to subscribe, while KNN is a black box that is slower to score.

---

## Repository Structure

```
.
├── lab_classification_bank_marketing_NoorZaid.ipynb   # Full analysis and answers
└── README.md
```

> `bank.csv` is not included. Download it from the [UCI repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) and place it where the notebook reads it (the notebook uses `/content/bank.csv`, the Google Colab path; change it if you run locally).

---

## How to Run

```bash
git clone https://github.com/NoorZaid/<your-repo-name>.git
cd <your-repo-name>
pip install numpy pandas matplotlib scikit-learn imbalanced-learn jupyter
jupyter notebook lab_classification_bank_marketing_NoorZaid.ipynb
```

---

## Tools and Libraries

Python, pandas, NumPy, Matplotlib, scikit-learn, imbalanced-learn (SMOTENC)

---

## Limitations and Next Steps

- Try tree-based models (Random Forest, Gradient Boosting) that handle mixed feature types well.
- Tune the Logistic Regression probability threshold using real call costs and subscription value.
- Use cross-validation instead of a single train/test split for more stable estimates.
- Because SMOTENC oversampled the training data, raw predicted probabilities are inflated and should be used for ranking, not read as true subscription rates.

---

## Author

**Noor Zaid** · [GitHub](https://github.com/NoorZaid)

## Data Source

Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository.
