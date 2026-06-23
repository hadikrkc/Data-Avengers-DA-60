# Machine Learning Analysis — Predicting Movie Revenue

**Team:** Data Avengers — DA-60
**Research Question:** Can we predict a film's box office revenue from pre-release features like budget, genre, and runtime?
**Problem Type:** Regression (revenue prediction) + Binary Classification (hit / flop)
**Models Used:** Linear Regression · Random Forest Regressor · Logistic Regression · Random Forest Classifier

---

## Table of Contents

1. [What Problem Are We Solving?](#1-what-problem-are-we-solving)
2. [How Did We Prepare the Data?](#2-how-did-we-prepare-the-data)
3. [Model 1 — Linear Regression](#3-model-1--linear-regression)
4. [Model 2 — Random Forest Regressor](#4-model-2--random-forest-regressor)
5. [How Do We Measure Performance?](#5-how-do-we-measure-performance)
6. [Model Results & Comparison](#6-model-results--comparison)
7. [Cross-Validation — A More Reliable Test](#7-cross-validation--a-more-reliable-test)
8. [Feature Importance — What Actually Drives Revenue?](#8-feature-importance--what-actually-drives-revenue)
9. [Residual Analysis — Where Do the Models Go Wrong?](#9-residual-analysis--where-do-the-models-go-wrong)
10. [Limitations](#10-limitations)
11. [Future Work](#11-future-work)
12. [Classification — Hit or Flop Prediction](#12-classification--hit-or-flop-prediction)
13. [Final Conclusion](#13-final-conclusion)

---

## 1. What Problem Are We Solving?

A movie studio about to greenlight a production needs to answer one fundamental question: *how much money will this film make?*

This is a **regression problem** — we are trying to predict a continuous number (box office revenue in USD), not just classify a film as "good" or "bad." The inputs are features that are known *before* a film is released, so in theory the model could be used to forecast earnings early in the production process.

### Why Regression, Not Classification?

It would be simpler to predict just "will this be a hit or a flop?" (a binary yes/no answer). But a regression model gives us more information: it estimates the actual dollar amount, which is far more useful for financial planning. If we later want classification, we can derive it from the regression output by applying a threshold.

### Data Used

| Property | Value |
|---|---|
| Source file | `movies_merged.csv` |
| Total rows | 5,280 films |
| Rows after removing nulls | 5,267 films |
| Training set | 4,214 films (80%) |
| Test set | 1,054 films (20%) |

---

## 2. How Did We Prepare the Data?

Before feeding data into a machine learning model, it must be cleaned and formatted correctly. This preparation phase is documented in **notebook `05_ml_preprocessing.ipynb`** and involves four steps.

### Step 1 — Feature Selection

We selected five input features (columns) that are available before a film's release:

| Feature | Type | Reason for Including |
|---|---|---|
| `budget` | Numeric | Strongest expected predictor; EDA showed r = 0.65 correlation with revenue |
| `runtime` | Numeric | Film length is a proxy for production scope and audience target |
| `release_year` | Numeric | Accounts for inflation and the overall growth of the box office over time |
| `primary_genre` | Categorical | Genre directly affects audience size (e.g., action vs. documentary) |
| `decade` | Categorical | Captures era-level trends (cinema economics changed dramatically decade by decade) |

The **target variable** (what we are trying to predict) is `revenue`.

Features like `vote_average` and `vote_count` were deliberately excluded because they are only available *after* a film has been seen by audiences — using them would be "cheating" and would make the model useless in practice.

### Step 2 — Encoding Categorical Variables

Machine learning algorithms work with numbers, not text. The `primary_genre` and `decade` columns contain text categories (e.g., "Action", "1990s"). We converted these into numbers using **One-Hot Encoding**:

> **What is One-Hot Encoding?**
> Each unique category becomes its own column, filled with 1 (yes, this film belongs to this category) or 0 (no, it does not). For example, a film with `primary_genre = "Action"` becomes `primary_genre_Action = 1`, `primary_genre_Comedy = 0`, etc.

After encoding, the dataset expanded from 6 columns to **33 columns** (32 features + 1 target).

### Step 3 — Train/Test Split

We split the dataset into two non-overlapping groups:

- **Training set (80% — 4,294 films):** Used to teach the model. The model sees both the input features and the correct answer (revenue) and learns from the patterns.
- **Test set (20% — 1,074 films):** The model has never seen these films. We use them to evaluate how well the model generalizes to new, unseen data.

We set `random_state=42` to ensure the split is the same every time the code is run, making results reproducible.

> **Why split at all?**
> If we trained and tested on the same data, the model could simply "memorize" the answers rather than learning real patterns. The test set simulates a real-world scenario where the model encounters films it has never seen before.

### Step 4 — Feature Scaling

The numeric features are on very different scales: `budget` can be in the hundreds of millions, while `runtime` is typically between 80 and 180. This mismatch can cause some models to give disproportionate weight to large-scale features.

We applied **StandardScaler**, which transforms each feature so it has a mean of 0 and a standard deviation of 1.

> **Important:** Scaling was applied only to the training set first, then the same transformation was applied to the test set. Fitting the scaler on the test set would "leak" test information into training and give misleadingly optimistic results. This is a critical discipline in machine learning.

Note: Scaling is only needed for **Linear Regression**. Random Forest, being a tree-based model, is completely unaffected by feature scales and was trained on the raw (unscaled) data.

---

## 3. Model 1 — Linear Regression

Documented in **notebook `06_ml_models.ipynb`**.

### What Is Linear Regression?

Linear Regression is the simplest and most interpretable machine learning model. It assumes that the relationship between the input features and the target variable can be expressed as a straight line (or, in multiple dimensions, a flat hyperplane).

In plain terms: the model learns a formula like this:

```
revenue = (w1 × budget) + (w2 × runtime) + (w3 × release_year) + ... + constant
```

Each weight (`w1`, `w2`, etc.) tells us how much each feature contributes to the prediction. A large positive weight means "when this feature increases, revenue increases proportionally."

### Why Did We Use It?

Linear Regression serves as the **baseline model**. In machine learning, it is standard practice to start with the simplest possible model before trying more complex ones. If a complex model doesn't significantly outperform a linear model, it signals one of two things:

1. The underlying relationship genuinely is linear (which is actually the case here — EDA showed budget and revenue correlate linearly at r = 0.65).
2. The complex model is overfitting rather than learning real patterns.

A baseline also gives us a reference point: "is the added complexity of Random Forest actually worth it?"

### Strengths

- Extremely fast to train
- Fully interpretable — each coefficient has a clear meaning
- No hyperparameters to tune
- Works well when the true relationship is approximately linear

### Weaknesses

- Cannot capture non-linear patterns (curved or stepped relationships)
- Sensitive to outliers (a handful of blockbusters with extreme revenue values can distort the learned weights)
- Requires feature scaling

### Results

| Metric | Value |
|---|---|
| MAE | $63.6 million |
| RMSE | $117.5 million |
| R² | 0.526 |

---

## 4. Model 2 — Random Forest Regressor

Documented in **notebook `06_ml_models.ipynb`**.

### What Is a Random Forest?

A Random Forest is an **ensemble model** — it combines the predictions of many individual decision trees to produce a more robust final answer.

**Step 1 — What is a Decision Tree?**
A decision tree makes predictions by asking a series of yes/no questions about the input features, branching left or right based on the answers, until it reaches a leaf node with a prediction. For example:

```
Is budget > $50M?
   ├── Yes → Is runtime > 120 min?
   │         ├── Yes → Predict $250M revenue
   │         └── No  → Predict $140M revenue
   └── No  → Is genre = "Action"?
             ├── Yes → Predict $90M revenue
             └── No  → Predict $30M revenue
```

A single decision tree is powerful but unreliable — it tends to overfit (memorize the training data rather than learning general patterns).

**Step 2 — What Makes It a "Forest"?**
A Random Forest trains **many trees** (we used 200), each on a different random subset of the training data and a random subset of features. The final prediction is the average of all 200 trees' predictions.

> **Why does averaging help?**
> Each tree makes different errors. When errors are random and uncorrelated, they cancel each other out when averaged. This is the same reason a group of 200 independent guesses averages closer to the truth than any single guess.

### Why Did We Use It?

1. **Non-linear relationships:** Random Forest can capture complex patterns that Linear Regression cannot. Even if budget and revenue are strongly linear, other features like genre or decade may interact in non-linear ways.
2. **Feature importance:** Random Forest provides a built-in measure of how much each feature contributes to the predictions. This was critical for answering our research question.
3. **Robustness to outliers:** Tree-based splits are less affected by extreme values than linear models.
4. **No scaling needed:** Features can be on any scale without affecting the model.

### Configuration

| Parameter | Value | Reason |
|---|---|---|
| `n_estimators` | 200 | More trees reduce variance; 200 was sufficient without being computationally expensive |
| `random_state` | 42 | Ensures reproducibility |
| `n_jobs` | -1 | Use all available CPU cores to train trees in parallel |

### Strengths

- Captures non-linear and interaction effects
- Robust to outliers and missing values
- Provides feature importance scores
- Less prone to overfitting than a single decision tree

### Weaknesses

- Less interpretable than Linear Regression (200 trees cannot be inspected individually)
- Slower to train
- Can still overfit if the forest learns noise in the training data

### Results

| Metric | Value |
|---|---|
| MAE | $64.4 million |
| RMSE | $120.7 million |
| R² | 0.500 |

---

## 5. How Do We Measure Performance?

We used three standard metrics for regression problems.

### MAE — Mean Absolute Error

The average of the absolute differences between predicted and actual revenue. If MAE = $60M, the model is off by $60 million on average.

> Easy to interpret: it's in the same units as the target (USD).

### RMSE — Root Mean Squared Error

Similar to MAE, but errors are squared before averaging (then square-rooted). This means **larger errors are penalized more heavily**.

> If a model makes a $300M mistake on one blockbuster, RMSE captures that worse than MAE does. For our dataset, where a few blockbusters have extreme revenues, RMSE is the more informative metric.

### R² — Coefficient of Determination

A value between 0 and 1 that tells us what proportion of the variation in revenue the model explains:

- **R² = 1.0** — perfect predictions
- **R² = 0.5** — the model explains 50% of the variation; the remaining 50% is not captured by the features
- **R² = 0.0** — the model is no better than simply predicting the mean revenue for every film
- **R² < 0** — the model is worse than predicting the mean (a sign something is very wrong)

---

## 6. Model Results & Comparison

Results from the held-out test set (1,074 films the models never saw during training):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | $63.6M | $117.5M | **0.526** |
| Random Forest | $64.4M | $120.7M | 0.500 |

**Linear Regression wins on the test set — and by a wider margin than expected.**

After removing data entry error records (budget/revenue < $10,000), Linear Regression outperforms Random Forest on all metrics. RMSE is $3.2M lower (2.7% improvement). This is consistent with the EDA finding (Pearson r = 0.65): the underlying relationship is predominantly linear. Random Forest's additional complexity was partially fitting noise introduced by the malformed records.

### What Does R² = 0.54 Mean in Practice?

Both models explain roughly 54% of the variation in revenue. The remaining 46% is driven by factors absent from our dataset:

- Marketing and advertising spend
- Cast recognition and star power
- Director's track record
- Release timing (holiday weekend vs. quiet January)
- Competition from other films opening the same week
- Critical reception and word of mouth

This is expected — no publicly available dataset contains all the variables that drive box office performance. An R² of 0.54 using only five input features is a reasonable result for a first-pass model.

---

## 7. Cross-Validation — A More Reliable Test

Documented in **notebook `07_ml_final.ipynb`**.

The test set results above are based on a single random split of the data. A different split might give slightly different numbers. **Cross-validation** gives a more robust estimate of model performance by repeating the evaluation multiple times.

### How Does 5-Fold Cross-Validation Work?

1. The training data is divided into 5 equal groups (folds).
2. The model is trained on 4 folds and tested on the remaining 1 fold.
3. This is repeated 5 times, each time using a different fold as the test set.
4. The 5 R² scores are averaged to get a more stable estimate of performance.

### Cross-Validation Results

| Model | CV R² (mean) | CV R² (std dev) |
|---|---|---|
| Linear Regression | **0.547** | **± 0.022** |
| Random Forest | 0.536 | ± 0.040 |

> **Cross-validation confirms Linear Regression as the superior model on the cleaned dataset.**

Linear Regression leads on both the test set and CV. More importantly, its CV standard deviation dropped from ±0.047 (uncleaned) to ±0.022 — the model is substantially more stable. Random Forest's CV R² also improved (0.498 → 0.536) but remains below Linear Regression on both metrics.

The lower CV R² for Random Forest, combined with its higher standard deviation, is a classic sign of **mild overfitting** — the model is learning some noise in the training data that does not generalize to new films.

### The Final Verdict

For reliable, stable predictions on unseen data, **Linear Regression is at least as strong a choice as Random Forest on this dataset.** Its simplicity is a virtue, not a limitation, when the underlying relationship is predominantly linear.

---

## 8. Feature Importance — What Actually Drives Revenue?

Documented in **notebooks `06_ml_models.ipynb` and `07_ml_final.ipynb`**.

Random Forest computes a feature importance score for each input — a measure of how much each feature contributed to reducing prediction error across all 200 trees.

### Results (Top Features)

| Feature | Importance | Interpretation |
|---|---|---|
| `budget` | **63.6%** | By far the dominant driver of revenue |
| `runtime` | 16.1% | Film length — second most influential |
| `release_year` | 9.5% | Accounts for growth of global box office over decades |
| All genre variables combined | ~8% | Genre matters, but far less than budget |
| All decade variables combined | ~3% | Weakest signal |

### What This Tells Us

The machine learning model independently confirms what our EDA found: **budget is the single most important predictor of revenue**, accounting for nearly two-thirds of all predictive power. This is a striking result — one number (the production budget) carries more signal than runtime, release era, and all genre information combined.

This makes intuitive sense: studios with large budgets can afford top-tier visual effects, globally recognized cast members, and — crucially — massive marketing campaigns that drive audiences to theaters.

The moderate importance of `runtime` (16.1%) is less obvious but interpretable: longer films tend to be larger-scope productions (epics, blockbusters), which correlates with both larger budgets and larger audiences. Genre carrying relatively little weight (~8%) suggests that within any genre, budget variation drowns out the genre effect.

---

## 9. Residual Analysis — Where Do the Models Go Wrong?

Documented in **notebook `07_ml_final.ipynb`**.

A **residual** is the difference between the actual revenue and the predicted revenue:

```
Residual = Actual Revenue − Predicted Revenue
```

- A positive residual means the model **underpredicted** (the film made more than expected)
- A negative residual means the model **overpredicted** (the film made less than expected)

Analyzing residuals reveals systematic patterns in the model's errors — things that the features we chose cannot explain.

### Key Residual Statistics (Random Forest)

| Statistic | Value |
|---|---|
| Mean residual | +$1.0 million |
| Standard deviation | $122.2 million |
| Skewness | 4.84 (strongly right-skewed) |
| Maximum underprediction | $1,876 million |
| Maximum overprediction | $571 million |

### What This Means

**Unbiased on average:** The mean residual of +$1M is practically zero, meaning the model is not systematically over- or under-predicting across the whole dataset. This is good.

**Large spread:** The standard deviation of $122M means individual predictions can be far off. This reflects the genuine difficulty of the problem — revenue is highly variable even among films with similar budgets.

**Extreme underprediction of blockbusters:** The maximum underprediction of $1,876M means the model missed the actual revenue of the biggest blockbuster in the dataset by nearly $2 billion. Films like the top-grossing superhero epics or franchises have revenues that are genuinely unpredictable from budget and genre alone — they require cultural momentum and franchise loyalty that no structured dataset captures.

**Right-skewed residuals:** The skewness of 4.84 confirms a systematic pattern: the model consistently underestimates the very highest-grossing films while making smaller errors for average films. This happens because extreme outliers pull the model's "mental average" down — the model has learned what a typical film earns, but it has no way of knowing when a film will transcend typical.

### Suggested Fix

Transforming the target variable using `log(revenue)` instead of raw revenue would compress the extreme outliers and likely reduce skewness significantly. This is a standard technique in regression when the target has a highly skewed distribution.

---

## 10. Limitations

### Data Limitations

- **Selection bias:** Only films with non-zero revenue and non-zero budget were included. This systematically excludes small independent films that never received wide theatrical distribution, skewing the dataset toward mainstream commercial productions.
- **Missing variables:** Marketing spend, cast and director historical performance, streaming rights value, and audience demographics are all known to affect revenue but are absent from the dataset.
- **Revenue definition:** Worldwide box office revenue is the target, but some films earn more through home media, streaming, and merchandise. Box office alone is an incomplete measure of commercial success.

### Model Limitations

- **Outlier sensitivity:** Both models struggle with the top ~1% of blockbusters. Their revenues are driven by franchise loyalty, viral marketing, and cultural moments — factors no structured feature can encode.
- **Linear assumptions:** Linear Regression assumes constant relationships (doubling the budget doubles the revenue). In reality, the relationship likely flattens at very high budgets.
- **Temporal leakage risk:** `release_year` as a feature could introduce subtle data leakage if the train/test split is not time-stratified. (Our split was random, not chronological.)
- **Generalizability:** The model was trained primarily on films from the 1990s–2010s era. Its predictions may be less reliable for films released after streaming fundamentally changed theater economics.

---

## 11. Future Work

Based on the model's residual patterns and known missing variables, several improvements could meaningfully increase performance:

| Improvement | Expected Benefit |
|---|---|
| Log-transform `revenue` as the target | Reduces right-skewed residuals; better behavior for outliers |
| Add cast/director performance features | Address the most important missing variable |
| Add marketing budget estimates (when available) | Budget alone ≠ total promotion spend |
| Sentiment analysis on film synopsis | Proxy for story quality and genre appeal |
| Time-stratified train/test split | Avoids temporal leakage and better simulates real-world use |
| Gradient Boosting models (XGBoost, LightGBM) | May outperform Random Forest on this data structure |
| Classification variant (ROI > 0 = profitable) | Natural next step — median-based hit/flop already implemented (see Section 12) |

---

## 12. Classification — Hit or Flop Prediction

Documented in **notebook `08_classification.ipynb`**.

### Problem Setup

While regression predicts the exact revenue amount, classification simplifies the answer to a binary label: **hit (1)** or **flop (0)**. This makes the output easier to communicate and directly actionable.

**Label construction:** The dataset median revenue ($30M) is used as the threshold.

```
hit  = 1   if revenue > $30M (median)
flop = 0   if revenue ≤ $30M (median)
```

By using the median, the dataset is perfectly balanced by construction — exactly 50% hits and 50% flops. No oversampling or class weighting is needed.

| Property | Value |
|---|---|
| Median threshold | $30 million |
| Hit films (1) | 2,679 (50.0%) |
| Flop films (0) | 2,689 (50.1%) |
| Training set | 4,294 films |
| Test set | 1,074 films (stratified) |

### Models Trained

**Logistic Regression (baseline)**
- The classification equivalent of Linear Regression
- Predicts the probability of a film being a hit; applies 0.5 threshold for the final label
- Requires feature scaling (StandardScaler applied)

**Random Forest Classifier**
- Same architecture as the Random Forest Regressor (200 trees, random_state=42)
- Outputs class probabilities for AUC calculation
- Does not require scaling

### Results

| Model | Accuracy | F1 Score | AUC-ROC |
|---|---|---|---|
| **Logistic Regression** | **0.773** | **0.750** | **0.860** |
| Random Forest Classifier | 0.746 | 0.737 | 0.828 |

**Logistic Regression wins on all three metrics** — mirroring the regression finding from Section 7. The predominantly linear relationship between budget and revenue means the simpler model generalizes better here too.

> **What does AUC-ROC = 0.86 mean?**
> If you randomly pick one hit film and one flop film from the dataset, the Logistic Regression model correctly identifies which is which 86% of the time. A random guess would achieve 50%. AUC of 0.86 is considered strong performance for a model using only five pre-release features.

### Cross-Validation Results

| Model | CV AUC (mean) | CV AUC (std dev) |
|---|---|---|
| Logistic Regression | 0.861 | ± 0.017 |
| Random Forest Classifier | 0.831 | ± 0.022 |

Cross-validation confirms Logistic Regression as the more stable model. The small standard deviations (±0.017 and ±0.022) indicate both models generalize consistently across different data splits.

### Feature Importance (Random Forest Classifier)

| Feature | Importance |
|---|---|
| `budget` | **47.2%** |
| `runtime` | 17.8% |
| `release_year` | 12.3% |
| All genre + decade variables | ~22.7% combined |

Budget remains the dominant predictor in classification as well — consistent with 63.6% importance in the regression model. The relative share is slightly lower here because binary classification compresses the signal (a $1B film and a $100M film both get label "1"), reducing the leverage of extreme budget values.

### Regression → Classification Bridge

A key insight: the regression model's revenue predictions can be thresholded to produce hit/flop labels — without training a separate classifier.

```
predicted_revenue = RandomForestRegressor.predict(X)
hit_prediction    = (predicted_revenue > $30M).astype(int)
```

This "bridge" achieves comparable accuracy to the dedicated classifier. A single regression model therefore serves double duty: it provides an exact revenue estimate *and* a hit/flop label, making it strictly more informative than a standalone classifier.

### Classification Limitations

- The $30M median threshold is dataset-dependent — a "hit" in this dataset may not match industry definitions
- A film earning $31M is labeled identically to one earning $2B; the label discards magnitude
- A more meaningful threshold could be **budget recovery** (`revenue > budget` = profitable), which directly maps to ROI > 0
- Post-release features (audience ratings, word of mouth) are deliberately excluded — including them would improve accuracy but destroy real-world applicability

---

## 13. Final Conclusion

We trained and compared two machine learning models to predict movie box office revenue using five pre-release features: budget, runtime, release year, primary genre, and decade.

**Summary of findings:**

| Question | Answer |
|---|---|
| Which model performed better on test data? | Linear Regression (R² = 0.526 vs RF 0.500) |
| Which model generalizes more reliably? | Linear Regression (CV R² = 0.547 ± 0.022 vs RF 0.536 ± 0.040) |
| What is the single most important predictor? | `budget` — 63.6% of feature importance |
| How accurate are the predictions? | On average ±$64M; large blockbusters can be off by billions |
| How much variance is explained? | ~53% (LR); the remaining ~47% comes from unobserved factors |

**The core finding:** Budget dominates. A film's production budget is, by far, the single strongest predictor of its box office revenue — more important than genre, runtime, or release era combined. This conclusion, already visible in the EDA correlation analysis (r = 0.65), is independently confirmed and quantified by the machine learning models.

**The model's honest boundary:** R² ≈ 0.54 means our features explain roughly half of what drives revenue. The other half — marketing, cast, cultural timing, word of mouth — is real and important, but it requires richer data sources to capture.

**Classification extension:** The binary hit/flop classifier (Section 12) achieves AUC-ROC = 0.86 with Logistic Regression — strong performance using the same five pre-release features. The regression model can also be thresholded to produce hit/flop labels, making it strictly more informative than a standalone classifier.

---

*Notebooks: `05_ml_preprocessing.ipynb` · `06_ml_models.ipynb` · `07_ml_final.ipynb` · `08_classification.ipynb`*
*Report generated: 2026-06-22*
