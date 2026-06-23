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

The strongest signal in the dataset. Pearson r = **0.650** on log-log scale (p ≈ 0); r² ≈ 0.42 — budget explains approximately 42% of revenue variance. Higher budget reliably predicts higher revenue, but scatter is wide: the same Blockbuster budget can produce anywhere from $31M to $2.79B in revenue. **Budget is a necessary but not sufficient condition for box office success.**

### Finding 2 — Budget Does Not Predict Quality

| Metric | Pearson r | Practical interpretation |
|---|---|---|
| Budget vs IMDb rating | **-0.094** | Near-zero: no meaningful relationship |
| Budget vs Tomatometer | **-0.221** | Moderate negative: more budget = lower critic scores |

Budget explains less than 0.3% of IMDb score variance. Higher-budget films score slightly lower with critics, likely because large studios favor commercial safety over creative risk.

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

| Model | MAE | RMSE | R² (test) | CV R² |
|---|---|---|---|---|
| Linear Regression | $63.6M | $117.5M | **0.526** | **0.547 ± 0.022** |
| Random Forest | $64.4M | $120.7M | 0.500 | 0.536 ± 0.040 |

**Key result:** After data cleaning, Linear Regression is the clear winner on both the test set and cross-validation. RF no longer outperforms LR on any metric. LR CV standard deviation dropped from ±0.047 to ±0.022 — the model is substantially more stable. This confirms that the underlying relationship is linear (EDA r = 0.65) and that the dirty data records were introducing noise that Random Forest partially over-fit to.

### Feature Importance (Random Forest)

| Feature | Importance |
|---|---|
| `budget` | **63.6%** |
| `runtime` | 16.1% |
| `release_year` | 9.5% |
| All genres + decades | ~10.8% combined |

Budget alone accounts for nearly two-thirds of the model's predictive power, confirming the EDA correlation finding from the ML angle.

### Residual Analysis

- Mean residual: +$1.0M (model is essentially unbiased)
- Residual std: $122.2M
- Residual skew: **4.84** (strong right skew — model underestimates blockbusters)
- Maximum underprediction: $1,876M

The model struggles most with extreme blockbusters. A log-transformed target would reduce this skew.

### Classification Results — Hit / Flop Prediction

**Task:** Predict whether a film earns above (hit) or below (flop) the dataset median revenue ($30M)  
**Label construction:** `hit = 1` if revenue > $30M median, `flop = 0` otherwise — balanced 50/50 split  
**Dataset:** 5,368 films → 4,294 train / 1,074 test (stratified 80/20, random_state=42)  
**Features:** Same 32 features as regression (no post-release data used)

| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Logistic Regression | **0.773** | **0.750** | **0.860** |
| Random Forest Classifier | 0.746 | 0.737 | 0.828 |

**Key result:** Logistic Regression outperforms Random Forest on all three metrics — again mirroring the regression finding. The predominantly linear relationship in the data means the simpler model generalizes better.

**Feature importance (RF Classifier):** `budget` = 47.2% — dominant, consistent with regression.

**Regression → Classification bridge:** Applying the $30M threshold to the regression model's revenue predictions yields comparable accuracy. A single regression model therefore serves double duty: it provides an exact revenue estimate *and* a hit/flop label without requiring a separate classifier.

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
- Marketing spend, cast, director track record, and release timing are absent from the data — these likely explain a significant portion of the unexplained ~46% revenue variance
- Rotten Tomatoes join via normalized title produced 81.3% coverage — 18.7% of films lack a critic score
- The ML models underpredict extreme blockbusters; log-transforming revenue would improve performance

---

## 7. Recommendations & Future Work

- **Log-transform revenue** for ML: reduces right-skew in residuals, likely improves R² by 5–10 points
- **Add marketing spend** as a feature: likely the single biggest missing variable
- **Classification variant (completed):** hit/flop prediction implemented in `08_classification.ipynb` using two thresholds: median revenue (balanced 50/50 split) and **revenue > budget** (profitability, implemented in Phase 10D)
- **Sentiment analysis** on film overviews/trailers: proxy for story quality
- **Director/cast historical performance**: proxy for creative execution quality

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
