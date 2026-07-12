# Final Report — Budget vs Success: Movie Data Analysis

**Team:** Data Avengers — DA-60  
**Research Question:** Does a bigger budget lead to better ratings and stronger box office results?  
**Dataset:** 5,280 films · 4 sources merged · 20 features  
**ML Deadline:** 11 July 2026

---

## 1. Introduction & Problem Statement

Every year, movie studios invest hundreds of millions of dollars into productions, betting that larger budgets will generate larger audiences and better reviews. Yet some of the most expensive films in history have flopped spectacularly, while low-budget independent films have become cultural phenomena.

This project tests that assumption with data. Using four merged datasets (Kaggle Movies Dataset, TMDB, IMDb, Rotten Tomatoes), we analyzed 5,381 films to quantify how production budget relates to:

- Box office revenue
- Audience ratings (IMDb)
- Critic scores (Rotten Tomatoes Tomatometer)
- Return on investment (ROI)

We also trained machine learning models to predict revenue from budget and other pre-release features (regression), and to classify films as commercial hits or flops (binary classification).

---

## 2. Data Sources & Cleaning Decisions

### Sources

| Dataset | Rows | Key Contribution |
|---|---|---|
| The Movies Dataset (Kaggle) | 45,466 → **5,280** usable | Budget, revenue, genres, imdb_id |
| TMDB 5000 Movies | 4,803 | Supplementary popularity score |
| IMDb title.ratings.tsv | 1,666,284 | IMDb rating (99.96% fill) |
| Rotten Tomatoes | 17,712 | Tomatometer (81.3% fill) |

**Usability filter:** Films with budget < $10,000 or revenue < $10,000 were excluded. Values below this threshold are data entry errors (budget stored as $1 or $100 instead of $1M), not real sub-$10K productions. This removed 101 records from the original 5,381 films.

| | Before cleaning | After cleaning |
|---|---|---|
| Films | 5,381 | **5,280** |
| r budget→revenue | 0.704 | **0.650** |
| r budget→IMDb | -0.079 | **-0.094** |
| r budget→Tomatometer | -0.176 | **-0.221** |
| Low tier median ROI | 180% | **189%** |
| RF MAPE | ~135,000% | **2,848%** |

The direction of all findings is unchanged. The MAPE reduction (135,000% → 2,848%) is the largest practical gain — previously, films with revenue = $100 inflated percentage errors to meaningless levels.

### Cleaning Decisions

| Decision | Reason |
|---|---|
| Log scale for all budget/revenue plots | Raw values span $1 → $2.79B — linear axis unreadable |
| `budget_tier` quartile cuts (Low/Mid/High/Blockbuster) | Enables group comparison beyond continuous correlation |
| `roi = (revenue - budget) / budget × 100` | Normalizes return regardless of absolute scale |
| IMDb join via `imdb_id` (exact match) | 99.96% fill — most reliable rating source |
| RT join via normalized title | 81.3% fill — acceptable for secondary metric |

### Final Dataset

- **Rows:** 5,280 films
- **Columns:** 20 (including derived: roi, budget_tier, decade, primary_genre)
- **Budget range:** $10,000 → $380M (median $17M)
- **Revenue range:** $10,000 → $2.79B (median $31M)
- **IMDb coverage:** 5,278 / 5,280 (99.96%)
- **Tomatometer coverage:** 4,326 / 5,280 (81.9%)

---

## 3. Key Findings

### Finding 1 — Budget Predicts Revenue (r = 0.65)

The strongest signal in the dataset. Pearson r = **0.650** on log-log scale (n = 5,280, p < .001); r² ≈ 0.42 — budget explains approximately 42% of revenue variance. Higher budget reliably predicts higher revenue, but scatter is wide: the same Blockbuster budget can produce anywhere from $31M to $2.79B in revenue. **Budget is a necessary but not sufficient condition for box office success.**

### Finding 2 — Budget Does Not Predict Quality

| Metric | Pearson r | n | p-value | Practical interpretation |
|---|---|---|---|---|
| Budget vs IMDb rating | **-0.094** | 5,278 | p < .001 | Statistically significant but practically negligible |
| Budget vs Tomatometer | **-0.221** | 4,326 | p < .001 | Moderate negative: more budget = lower critic scores |

Budget explains less than 1% of IMDb score variance. Even though the IMDb relationship is statistically significant (large sample size makes even tiny effects detectable), it is too small to be practically meaningful. Higher-budget films score slightly lower with critics, likely because large studios favor commercial safety over creative risk.

### Finding 3 — Low Budget Delivers the Best ROI

| Tier | Median ROI | Avg IMDb |
|---|---|---|
| **Low** | **189%** | **6.67** |
| Mid | 80% | 6.52 |
| High | 70% | 6.38 |
| Blockbuster | 116% | 6.46 |

Low-budget films are the most efficient investment by ROI and score highest with audiences. The Mid–High range ($20M–$80M) carries the highest financial risk with no quality advantage.

### Finding 4 — Notable Exceptions Define the Pattern

**Cheap hits (Low tier):**  
- *Blair Witch Project*: ~$60K budget → $248M revenue  
- *Paranormal Activity*: minimal budget → $193M revenue  
- Horror and Drama dominate; creative necessity drives innovation.

**Expensive flops (Blockbuster tier):**  
- *Foodfight!*: $65M → $0.1M (ROI: **-100%**)  
- *The Adventures of Pluto Nash*: $100M → $7.1M (ROI: **-93%**)  
- High spend with poor execution is catastrophic.

### Finding 5 — Budget & Revenue Trend by Decade

Both metrics grew consistently across the century. Median budget peaked at $25M in the 2000s; median revenue at $38.6M. In the 2010s, budget dipped to $19.8M while revenue held at $36.9M — a widening profit margin possibly driven by streaming economics. Revenue exceeded budget in every decade: the industry is median-profitable across all eras.

### Finding 6 — Genre Stratifies Budget

Highest-budget genres: Animation ($59.5M median), Adventure ($40M), Family ($35M). Lowest-budget genres: Horror, Documentary — which explains why these genres deliver the highest ROI.

---

## 4. Machine Learning Results

**Task:** Predict revenue from pre-release features (budget, runtime, release_year, primary_genre)  
**Dataset:** 5,268 films → **chronological split**: pre-2010 films = train, 2010+ = test  
**Features:** ~21 (3 numeric + ~18 one-hot encoded genre dummies; `decade` excluded — redundant with `release_year`)  
**Primary model (Film Predictor):** Random Forest on log1p(revenue) target, back-transformed with expm1()

### Model Comparison

| Model | MAE | RMSE | R² (test) | CV R² (train) |
|---|---|---|---|---|
| Linear Regression | $68.7M | $136.9M | **0.597** | 0.403 ± 0.123 |
| Random Forest | $73.5M | $137.6M | 0.593 | 0.369 ± 0.131 |

**Key result:** With chronological split (pre-2010 train / 2010+ test), both models achieve R² ≈ 0.60 — higher than the previous random-split result (0.53). Linear Regression edges out Random Forest on the test set. RMSE is higher in absolute terms ($137M vs $117M before) because 2010+ films have higher revenues on average — the model is being tested on a harder distribution. LR CV R² = 0.403 ± 0.123 (Pipeline fix applied — scaling within each fold via sklearn Pipeline prevents numerical overflow).

### Feature Importance (Random Forest)

| Feature | Importance |
|---|---|
| `budget` | **56.6%** |
| `runtime` | 21.6% |
| `release_year` | 10.9% |
| All genres combined | ~11.0% |

Budget remains the dominant predictor. Runtime's importance increased (21.6% vs 16.1%) after removing `decade` dummies, which were absorbing some temporal variance previously.

### Residual Analysis

- Mean residual: +$3.8M (near-zero — model is approximately unbiased)
- Residual std: $137.6M
- Residual skew: **2.13** (much lower than before — 2010+ test set has fewer ultra-outliers)
- Maximum underprediction: $1,109M

The model struggles most with extreme blockbusters. A log-transformed target would reduce this skew.

### Classification Results — Hit / Flop Prediction

**Task:** Predict whether a film earns above (hit) or below (flop) the training-set median revenue  
**Label construction:** `hit = 1` if revenue > training-set median ($29.2M), `flop = 0` otherwise  
**Dataset:** 5,268 films → chronological split: 3,664 train (pre-2010) / 1,604 test (2010+)  
**Features:** `budget`, `runtime`, `release_year`, `primary_genre` (same as regression, no `decade`)

| Model | Accuracy | F1 | AUC-ROC | CV AUC |
|---|---|---|---|---|
| Logistic Regression | **0.787** | **0.793** | **0.874** | 0.818 ± 0.040 |
| Random Forest Classifier | 0.771 | 0.779 | 0.850 | 0.783 ± 0.012 |

**Key result:** Logistic Regression outperforms Random Forest on all metrics — mirroring the regression finding. Chronological split yields better metrics than the previous random split (LR AUC: 0.874 vs 0.860 before), suggesting the pre-2010 training data captures budget→revenue patterns that generalize well to 2010+ films.

**Profitability threshold (revenue > budget):** LR Accuracy 0.690, AUC-ROC 0.652. Predicting break-even is harder than predicting median-outperformance — budget alone is less predictive of whether a film recoups its cost.

**Regression → Classification bridge:** Applying the training median threshold to the regression model's revenue predictions yields comparable accuracy. A single regression model therefore serves double duty: it provides an exact revenue estimate *and* a hit/flop label without requiring a separate classifier.

---

## 5. Answer: Does Budget Drive Success?

| Dimension | Answer |
|---|---|
| **Box office revenue** | **Yes** — strong correlation (r = 0.65, r² ≈ 0.42); ML confirms (budget = dominant feature, R² ≈ 0.5) |
| **Audience rating (IMDb)** | **No** — near-zero correlation (r = -0.094) |
| **Critic score (Tomatometer)** | **Negative** — more budget correlates with lower critic scores (r = -0.221) |
| **Return on investment** | **Counterintuitively no** — Low tier delivers best ROI (189%) |

**Overall:** A high budget increases the *scale* of a film's release and its expected revenue, but has no positive effect on quality perception and actually produces worse ROI than low-budget filmmaking. The most profitable strategy by ROI is low-budget genre production (particularly Horror); the most reliable large-scale strategy is franchise-driven Blockbuster production. The mid-range ($20M–$80M) carries the highest financial risk with no distinct quality or ROI advantage.

---

## 6. Limitations

- Dataset only includes films with budget ≥ $10,000 and revenue ≥ $10,000 — films without public financial data and records with data entry errors are excluded (selection bias toward mainstream productions)
- Dataset coverage ends at 2017 — post-2017 streaming economics, franchise dominance, and the COVID-19 box office collapse (2020–2021) are not reflected
- Marketing spend, cast, director track record, and release timing are absent from the data — these likely explain a significant portion of the unexplained ~46% revenue variance
- Rotten Tomatoes join via normalized title produced 81.3% coverage — 18.7% of films lack a critic score
- Budget and revenue figures are not inflation-adjusted — a $50M budget in 1995 has a different real value than $50M in 2015, which affects decade-trend comparisons
- ML models underpredict extreme blockbusters (max underprediction: $1,109M); the right-skewed revenue distribution limits accuracy at the high end even with log-transformation

---

## 7. Recommendations & Future Work

### Completed Improvements
- **Log-transform revenue** *(implemented)*: RF model trained on `log1p(revenue)` with `expm1()` back-transform — reduces residual skew from 4.84 to 2.13
- **Chronological train/test split** *(implemented)*: pre-2010 train / 2010+ test eliminates future-data leakage — R² improved from 0.531 to 0.597
- **Profitability classification** *(implemented)*: second threshold `revenue > budget` added alongside median threshold in `08_classification.ipynb` — LR AUC 0.652

### Near-Term Improvements (Faz 2)

- **Inflation adjustment**: Apply CPI multipliers (2000 base year) to convert nominal budget/revenue to real values. A $50M budget in 1990 ≈ $98M in 2017 dollars. This would make decade-trend comparisons significantly more reliable and is straightforward to implement with a lookup table.

- **Interaction features**: Add derived features such as `budget_per_minute` (production intensity) and genre × budget tier combinations. These may expose non-linear patterns that neither budget nor genre alone captures, and could shift feature importance away from the dominant `budget` signal.

- **Prediction intervals (quantile regression)**: Replace point estimates with confidence ranges (e.g., "80% probability the film earns $15M–$120M"). Implementable with `GradientBoostingRegressor(loss='quantile')`. More informative for decision-making than a single predicted value.

### Longer-Term Improvements (Faz 3)

- **Extend dataset to 2024 via TMDB API**: The free TMDB API provides budget/revenue data for recent films. Estimated yield: ~2,000–4,000 films with financial data for 2017–2024. Note: 2020–2021 should be handled separately or flagged — the COVID-19 pandemic reduced global box office by ~80% and would distort any trend analysis if included without context.

- **Add marketing spend**: The single biggest missing variable. No public dataset covers marketing budgets systematically, but studio-reported P&A (prints & advertising) figures exist for major releases through trade publications (Deadline, Box Office Mojo).

- **Cast and director track record**: Historical performance of lead actors and directors as features — a strong proxy for creative execution quality and audience draw. Requires IMDb filmography data (available via IMDb Non-Commercial Datasets).

- **Sentiment analysis on plot overviews**: NLP-based quality proxy from film synopsis text — classifiable as `positive/neutral/negative` using a pre-trained model (e.g., distilBERT). May partially explain the critic score variance unexplained by budget.

---

## 8. Notebooks & Scripts

| File | Description |
|---|---|
| `Source/notebooks/01_data_overview.ipynb` | Initial data inspection |
| `Source/notebooks/02_cleaning_and_merge.ipynb` | Cleaning pipeline → movies_merged.csv |
| `Source/notebooks/03_eda.ipynb` | Full EDA with 11 plots |
| `Source/notebooks/04_final_analysis.ipynb` | Presentation-quality key findings |
| `Source/notebooks/05_ml_preprocessing.ipynb` | Feature engineering & train/test split |
| `Source/notebooks/06_ml_models.ipynb` | LR vs RF training & comparison |
| `Source/notebooks/07_ml_final.ipynb` | CV, residual analysis, ML conclusion |
| `Source/notebooks/08_classification.ipynb` | Hit/flop binary classification (LR + RF Classifier) |
| `Source/scripts/helpers.py` | Path constants |
| `Source/scripts/load_data.py` | Dataset loaders |
| `Source/scripts/clean_data.py` | 8 cleaning functions |
| `Source/scripts/merge_data.py` | build_master_dataset() |
