# Project Proposal

**Project Title** - Budget vs Success: Does a Bigger Budget Lead to Better Ratings and Stronger Box Office Results?

## Problem Statement

Every year, movie studios pour heavy amounts of money into productions, betting that bigger budgets will draw bigger crowds and better reviews. But anyone who follows movies knows that's not always how it plays out. Some of the most expensive films in history have flopped, while smaller productions made on a fraction of the budget have gone on to become cultural phenomena.
So the question we want to dig into is simple: does spending more actually lead to better outcomes? We'll look at this through real data, examining how production budgets relate to box office earnings, audience ratings, and critic scores across hundreds of films. We'll also explore whether the answer changes depending on the genre or the era the film was made in, because the industry today looks very different from even a decade ago.
Using four combined datasets from sources like Kaggle, TMDB, Rotten Tomatoes, and IMDb, we'll build a data pipeline to bring this all together and find out what the numbers actually say.

## GitHub Setup

The GitHub repository has been created (https://github.com/hadikrkc/Data-Avengers-DA-60) and will be used to manage files, code, updates, and overall project work.

## Microsoft Teams Setup

The Microsoft Teams group has been set up ( Data Avengers - DA Project Group | Group Chat | Microsoft Teams ) and will be used for communication, meetings, file sharing, and coordination among team members.

## Introduction Brief

The movie industry is one of the most competitive industries, where budgets can vary from small independent films to very large blockbuster productions. In many cases, studios believe that spending more on cast, effects, and promotion increases the chances of success. This project aims to test that idea through data analysis. We will compare movie budgets with box office revenue and ratings to understand whether bigger spending actually leads to better outcomes.

## Team Objective

Our goal is to collect a relevant movie dataset, analyze the data, create useful visualizations, and present clear findings on the relationship between movie budget and success.

## Results Summary

| Dimension | Finding |
|---|---|
| Budget → Revenue | Strong positive correlation — r = **0.70**, r² ≈ 0.50 |
| Budget → IMDb Rating | Near-zero — r = -0.057; budget does not buy quality |
| Budget → Tomatometer | Slightly negative — r = -0.176; more budget, marginally worse critic scores |
| Best ROI tier | **Low budget** — median ROI 180% vs 69% for High tier |

**Short answer:** Budget predicts revenue but not quality. Low-budget films deliver better ROI and equal or higher audience ratings.

---

## Project Structure

```
Data-Avengers-DA-60/
├── Data/
│   ├── raw/            # Source datasets (git-ignored)
│   └── processed/      # movies_merged.csv (5,381 rows, 20 columns)
├── Source/
│   ├── notebooks/      # 01–08 Jupyter notebooks
│   ├── scripts/        # helpers.py, load_data.py, clean_data.py, merge_data.py
│   └── streamlit_app.py
├── Output/
│   ├── figures/        # PNG plots exported from notebooks
│   ├── exports/        # JSON/CSV for Power BI, Tableau, and Streamlit
│   └── models/         # trained model .pkl files (git-ignored)
├── Documentation/
│   └── reports/        # data_overview.md, eda_summary.md, final_report.md, ml_analysis.md
├── requirements.txt
└── README.md
```

## Running the Project

```bash
pip install -r requirements.txt
jupyter lab
```

Run notebooks in order: `01 → 02 → 03 → 04 → 05 → 06 → 07 → 08`

After running each notebook, execute the **Export** cell at the bottom to generate `Output/figures/` and `Output/exports/`.

## Streamlit Dashboard

```bash
streamlit run Source/streamlit_app.py
```

Five pages:
- **Overview** — key findings and ML summary table
- **EDA** — interactive budget/revenue/ROI/genre/decade charts
- **ML Regression** — feature importance, actual vs predicted, residual analysis
- **ML Classification** — ROC curve, confusion matrices, metric comparison
- **Film Predictor** — enter budget/runtime/genre to get a revenue prediction and hit/flop probability

## System Design & Workflow

1. **Data Collection** — 4 Kaggle datasets downloaded to `Data/raw/`
2. **Data Cleaning & Preprocessing** — `clean_data.py`: 8 functions, zero-budget filter, feature engineering
3. **Dataset Merge** — `merge_data.py`: IMDb join via imdb_id (99.96%), RT join via normalized title (81.3%)
4. **Exploratory Data Analysis** — `03_eda.ipynb`: 11 plots, correlations, ROI by tier, decade trends
5. **Final Visualization** — `04_final_analysis.ipynb`: 4 key findings + conclusion
6. **Machine Learning** — `05_ml_preprocessing.ipynb`: feature engineering, encoding, train/test split; `06_ml_models.ipynb`: Linear Regression (R²=0.531) vs Random Forest Regressor (R²=0.542); `07_ml_final.ipynb`: cross-validation, residual analysis; `08_classification.ipynb`: binary hit/flop classification, Logistic Regression AUC-ROC=0.860
7. **Reporting** — `Documentation/reports/final_report.md`, `Documentation/reports/ml_analysis.md`

## Documentation

- `Documentation/reports/problem_analysis.md` — problem definition, research questions, and scope
- `Documentation/reports/data_overview.md` — dataset inspection findings and full column lineage
- `Documentation/reports/eda_summary.md` — EDA key numbers and interpretations
- `Documentation/reports/final_report.md` — complete project report
- `Documentation/reports/ml_analysis.md` — detailed ML analysis: regression, classification, cross-validation, residual analysis

## Dataset Sources

Raw datasets are stored locally under `Data/raw/` and excluded from Git tracking.

- `The Movies Dataset`: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
- `TMDB 5000 Movies Dataset`: https://www.kaggle.com/datasets/muhammadnaumank/tmdb-5000-movies-dataset
- `Rotten Tomatoes movies and critic reviews dataset`: https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset
- `IMDb Dataset`: https://www.kaggle.com/datasets/ashirwadsangwan/imdb-dataset

Recommended usage:

- Keep downloaded raw files in `Data/raw/`.
- Keep cleaned, merged, or renamed outputs in `Data/processed/`.
- Do not edit raw source files in place unless there is a strong reason.
- If folder or file names are standardized later, update all notebook and script paths accordingly.
