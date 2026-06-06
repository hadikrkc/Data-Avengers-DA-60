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