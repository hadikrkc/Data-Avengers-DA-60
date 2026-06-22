# Project Progress Log

---

## Week 1 — May 3, 2026
**Focus:** Project kickoff and repository setup

- Repository created on GitHub: https://github.com/hadikrkc/Data-Avengers-DA-60
- Initial project folder structure established
- Project responsibilities assigned among team members
- System Design & Workflow defined
- First commit pushed

---

## Week 2 — May 10, 2026
**Focus:** Research and dataset selection

- Research on available movie datasets on Kaggle
- Evaluated dataset suitability (coverage, join keys, column quality)
- Selected four datasets: The Movies Dataset, TMDB 5000, IMDb ratings, Rotten Tomatoes
- Downloaded raw files to `Data/raw/` (git-ignored)
- Documented dataset sources in `datasetsLinksCameFrom.txt`

---

## Week 3 — May 16–17, 2026
**Focus:** Problem definition and project structure

- Wrote `Documentation/reports/problem_analysis.md` — research questions, scope, hypotheses
- Built initial project folder structure (`Source/`, `Data/`, `Documentation/`, `Logs/`)
- Updated README with problem statement, team objectives, and dataset setup
- Committed: `Added problem analysis document and updated README file`

---

## Week 4 — May 23, 2026
**Focus:** Repository hygiene and documentation

- Added `.gitignore` to exclude `Data/raw/` and `Data/processed/` from version control
- Finalized README with GitHub and Teams setup sections
- Committed: `Add gitignore file and ReadMe file`

---

## Week 5 — May 29–30, 2026
**Focus:** Full data pipeline, EDA, machine learning, and reporting

### Data Pipeline (Session 1 & 2)
- Created `requirements.txt` (pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter)
- Wrote all four scripts: `helpers.py`, `load_data.py`, `clean_data.py`, `merge_data.py`
  - `clean_data.py`: 8 functions — normalize_title, coerce_numeric, filter_positive_budget_revenue, extract_year, extract_primary_genre, add_roi_column, add_budget_tier, add_decade
  - `merge_data.py`: build_master_dataset() — IMDb join via imdb_id (99.96%), RT join via normalized title (81.3%)
- Wrote and ran `01_data_overview.ipynb` — initial dataset inspection
- Wrote and ran `02_cleaning_and_merge.ipynb` — produced `Data/processed/movies_merged.csv`
  - 5,381 rows · 20 columns
- Wrote notebook content for `03_eda.ipynb` through `07_ml_final.ipynb`
- Committed: `Add requirements file for python`, `load data and overview`, `Clean and Merge data`

### EDA & Final Analysis (Session 3)
- Ran `03_eda.ipynb` — 11 plots, all interpretation cells filled
  - Budget vs Revenue: Pearson r = 0.704
  - Budget vs IMDb: r = -0.057 (no relationship)
  - Budget vs Tomatometer: r = -0.176 (slight negative)
  - Low-budget median ROI: 180% vs High-tier 69%
- Ran `04_final_analysis.ipynb` — 4 key findings + conclusion written
- Wrote `Documentation/reports/eda_summary.md`

### Machine Learning (Session 3)
- Ran `05_ml_preprocessing.ipynb` — 5,368 rows, 32 features, 80/20 split
- Ran `06_ml_models.ipynb`
  - Linear Regression: MAE $62.0M, RMSE $123.5M, R² = 0.531
  - Random Forest: MAE $60.8M, RMSE $122.1M, R² = 0.542
  - Feature importance: `budget` = 63.6%
- Ran `07_ml_final.ipynb` — cross-validation, residual analysis, ML conclusion
  - CV result: LR (0.536 ± 0.047) outperforms RF (0.498 ± 0.078) — RF shows mild overfitting
- Wrote `Documentation/reports/final_report.md`
- Updated `README.md` with results summary, project structure, and running instructions

---

## Week 6 — June 22, 2026
**Focus:** Classification extension and comparative analysis

### Classification Model
- Created `08_classification.ipynb` — binary hit/flop prediction using median revenue ($30M) as threshold
  - Logistic Regression: Accuracy 0.773, F1 0.750, AUC-ROC 0.860
  - Random Forest Classifier: Accuracy 0.746, F1 0.737, AUC-ROC 0.828
  - Logistic Regression outperforms RF — consistent with regression findings (linear relationship dominates)
  - Top feature: `budget` (47.2% importance) — consistent with regression model (63.6%)
  - Added confusion matrices, ROC curve, cross-validation, and regression→classification bridge
- Dataset: 5,368 films, 2,679 hits / 2,689 flops (balanced by construction)

### Documentation
- Created `Documentation/reports/ml_analysis.md` — detailed ML analysis report (13 sections)
  - Covers preprocessing, both regression models, classification models, residual analysis
  - Section 12: comparative analysis vs TMDB classification approach (data leakage risk identified)
- Created `Documentation/reports/ml_analiz_TR.md` — Turkish translation of ml_analysis.md
- Updated `final_report.md` — added Section 4b (classification results) and notebook 08 entry
- Updated `Documentation/reports/problem_analysis.md` — added classification to analytical approach

---

## Upcoming — Before July 4, 2026
**Focus:** ML presentation preparation

- [ ] Prepare slides: Problem → Dataset → Models → Evaluation → Results → Comparison → Conclusion
- [ ] All group members review every section (Q&A preparation)
- [ ] Submit notebook + slides at least 2 days before presentation
