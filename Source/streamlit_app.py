"""
Budget vs Success — Interactive Dashboard
Run with: streamlit run Source/streamlit_app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import json
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
FIGURES  = PROJECT_ROOT / 'Output' / 'figures'
EXPORTS  = PROJECT_ROOT / 'Output' / 'exports'
MODELS   = PROJECT_ROOT / 'Output' / 'models'
DATA_DIR = PROJECT_ROOT / 'Data' / 'processed'

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_df():
    path = DATA_DIR / 'movies_merged.csv'
    if not path.exists():
        return None
    return pd.read_csv(path)

@st.cache_data
def load_json(name):
    path = EXPORTS / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))

def load_image(name):
    path = FIGURES / name
    if not path.exists():
        return None
    return Image.open(path)

def load_model(name):
    path = MODELS / name
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

def exports_ready():
    return (EXPORTS / 'model_results.json').exists()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Budget vs Success',
    page_icon='🎬',
    layout='wide',
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title('🎬 Budget vs Success')
st.sidebar.caption('Data Avengers — DA-60')

pages = [
    '📊 Overview',
    '🔍 EDA — Exploratory Analysis',
    '🤖 ML — Regression',
    '🏷️ ML — Classification',
    '🎬 Film Predictor',
    '📁 Dataset',
]
page = st.sidebar.radio('Navigate', pages)

if not exports_ready():
    st.sidebar.warning(
        'Output files not found. Run notebooks 03 → 04 → 06 → 07 → 08 '
        'and execute the Export cell in each notebook first.'
    )

st.sidebar.markdown('---')
st.sidebar.markdown('**Dataset:** 5,381 films · 20 features')
st.sidebar.markdown('**Data Sources**')
st.sidebar.markdown(
    '- [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)\n'
    '- [TMDB 5000 Movies](https://www.kaggle.com/datasets/muhammadnaumank/tmdb-5000-movies-dataset)\n'
    '- [Rotten Tomatoes](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset)\n'
    '- [IMDb Dataset](https://www.kaggle.com/datasets/ashirwadsangwan/imdb-dataset)'
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == '📊 Overview':
    st.title('Budget vs Success')
    st.subheader('Does a bigger budget lead to better ratings and stronger box office results?')

    st.markdown('---')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Films Analyzed', '5,381')
    col2.metric('Budget → Revenue (r)', '0.70')
    col3.metric('Budget → IMDb (r)', '-0.057')
    col4.metric('Low-Budget Median ROI', '180%')

    st.markdown('---')

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('### Key Findings')
        st.markdown("""
| Dimension | Finding |
|---|---|
| Budget → Revenue | Strong positive — r = **0.70**, r² ≈ 0.50 |
| Budget → IMDb Rating | Near-zero — r = **-0.057**; budget does not buy quality |
| Budget → Tomatometer | Slightly negative — r = **-0.176** |
| Best ROI tier | **Low budget** — median ROI 180% vs 69% for High tier |

**Short answer:** Budget predicts revenue but not quality.
Low-budget films deliver better ROI and equal or higher audience ratings.
        """)

    with col_r:
        img = load_image('budget_vs_revenue_final.png')
        if img:
            st.image(img, caption='Budget vs Revenue — colored by budget tier', use_container_width=True)
        else:
            st.info('Run notebook 04 Export cell to generate this figure.')

    st.markdown('---')
    st.markdown('### ML Model Summary')
    ml_col1, ml_col2 = st.columns(2)
    with ml_col1:
        st.markdown('**Regression — Revenue Prediction**')
        st.markdown("""
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | $62.0M | $123.5M | 0.531 |
| Random Forest | $60.8M | $122.1M | 0.542 |

CV winner: **Linear Regression** — R² 0.536 ± 0.047
        """)
    with ml_col2:
        st.markdown('**Classification — Hit / Flop Prediction**')
        st.markdown("""
| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Logistic Regression | 0.773 | 0.750 | **0.860** |
| Random Forest | 0.746 | 0.737 | 0.828 |

Median threshold: **$30M**
        """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == '🔍 EDA — Exploratory Analysis':
    st.title('Exploratory Data Analysis')

    df = load_df()

    tab1, tab2, tab3, tab4 = st.tabs([
        'Budget vs Revenue', 'ROI by Tier', 'Genre Analysis', 'Decade Trends'
    ])

    with tab1:
        st.subheader('Budget vs Revenue')
        if df is not None:
            _sub = df[['budget', 'revenue', 'budget_tier']].dropna()
            fig = px.scatter(
                _sub,
                x='budget', y='revenue',
                color='budget_tier',
                log_x=True, log_y=True,
                opacity=0.4,
                labels={'budget': 'Budget (USD)', 'revenue': 'Revenue (USD)', 'budget_tier': 'Tier'},
                title='Budget vs Revenue (log scale) — Pearson r = 0.70',
                color_discrete_map={
                    'Low': '#4e9af1', 'Mid': '#f4a261',
                    'High': '#e76f51', 'Blockbuster': '#2d6a4f'
                },
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            img = load_image('budget_vs_revenue_final.png')
            if img:
                st.image(img, use_container_width=True)
            else:
                st.info('Dataset not found. Run notebook 02 first or export figures from notebook 04.')
        st.markdown('**r = 0.70** — budget explains ~50% of revenue variance. Wide scatter means budget is necessary but not sufficient.')

    with tab2:
        st.subheader('ROI by Budget Tier')
        tier_data = load_json('budget_tier_stats.json')
        if tier_data:
            tier_df = pd.DataFrame(tier_data)
            fig = px.bar(
                tier_df, x='budget_tier', y='median_roi',
                color='budget_tier',
                labels={'budget_tier': 'Budget Tier', 'median_roi': 'Median ROI (%)'},
                title='Median ROI by Budget Tier',
                color_discrete_map={
                    'Low': '#4e9af1', 'Mid': '#f4a261',
                    'High': '#e76f51', 'Blockbuster': '#2d6a4f'
                },
                text='median_roi',
            )
            fig.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            img = load_image('roi_by_tier.png')
            if img:
                st.image(img, use_container_width=True)
            else:
                st.info('Run notebook 03 Export cell first.')
        st.markdown('Low-budget films: **180% median ROI**. Blockbuster: 116%. Mid–High tier ($20M–$80M) is the riskiest zone.')

    with tab3:
        st.subheader('Genre Analysis')
        genre_data = load_json('genre_stats.json')
        if genre_data:
            genre_df = pd.DataFrame(genre_data).sort_values('median_budget', ascending=False).head(15)
            fig = px.bar(
                genre_df, x='primary_genre', y='median_budget',
                labels={'primary_genre': 'Genre', 'median_budget': 'Median Budget (USD)'},
                title='Median Budget by Genre (Top 15)',
                color='median_roi',
                color_continuous_scale='RdYlGn',
                hover_data=['count', 'median_roi', 'avg_imdb'],
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('Color = Median ROI. Animation ($59.5M) and Adventure ($40M) have the highest budgets; Horror and Documentary deliver the best ROI.')
        else:
            img = load_image('genre_budget.png')
            if img:
                st.image(img, use_container_width=True)
            else:
                st.info('Run notebook 03 Export cell first.')

    with tab4:
        st.subheader('Budget & Revenue by Decade')
        decade_data = load_json('decade_trends.json')
        if decade_data:
            dec_df = pd.DataFrame(decade_data)
            dec_df['decade'] = pd.to_numeric(
                dec_df['decade'].astype(str).str.replace('s', '', regex=False), errors='coerce'
            ).astype('Int64')
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dec_df['decade'], y=dec_df['median_budget'] / 1e6,
                mode='lines+markers', name='Median Budget (M USD)',
                line=dict(color='steelblue', width=2), marker=dict(size=8)
            ))
            fig.add_trace(go.Scatter(
                x=dec_df['decade'], y=dec_df['median_revenue'] / 1e6,
                mode='lines+markers', name='Median Revenue (M USD)',
                line=dict(color='coral', width=2), marker=dict(size=8)
            ))
            fig.update_layout(
                title='Budget and Revenue Trend by Decade',
                xaxis_title='Decade', yaxis_title='Million USD',
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            img = load_image('decade_trends_final.png')
            if img:
                st.image(img, use_container_width=True)
            else:
                st.info('Run notebook 03 Export cell first.')
        st.markdown('Revenue consistently exceeded budget in every decade. 2010s: budget fell to $19.8M while revenue held at $36.9M.')

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ML Regression
# ══════════════════════════════════════════════════════════════════════════════
elif page == '🤖 ML — Regression':
    st.title('Machine Learning — Revenue Regression')
    st.markdown('Models trained to predict exact box office revenue using pre-release features only.')

    model_data = load_json('model_results.json')

    if model_data and 'regression' in model_data:
        reg = model_data['regression']
        lr_r = reg.get('linear_regression', {})
        rf_r = reg.get('random_forest', {})

        st.subheader('Model Performance')
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('**Linear Regression**')
            st.metric('R²', f"{lr_r.get('R2', 'N/A')}")
            st.metric('MAE', f"${lr_r.get('MAE_M', 'N/A'):.1f}M")
            st.metric('RMSE', f"${lr_r.get('RMSE_M', 'N/A'):.1f}M")
            if 'cv_r2_mean' in lr_r:
                st.metric('CV R²', f"{lr_r['cv_r2_mean']:.4f} ± {lr_r['cv_r2_std']:.4f}")
        with c2:
            st.markdown('**Random Forest Regressor**')
            st.metric('R²', f"{rf_r.get('R2', 'N/A')}")
            st.metric('MAE', f"${rf_r.get('MAE_M', 'N/A'):.1f}M")
            st.metric('RMSE', f"${rf_r.get('RMSE_M', 'N/A'):.1f}M")
            if 'cv_r2_mean' in rf_r:
                st.metric('CV R²', f"{rf_r['cv_r2_mean']:.4f} ± {rf_r['cv_r2_std']:.4f}")

        if 'residual_analysis' in reg:
            res = reg['residual_analysis']
            st.markdown('---')
            st.subheader('Residual Analysis (Random Forest)')
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric('Mean Residual', f"${res.get('mean_M', 'N/A'):.1f}M")
            rc2.metric('Std Dev', f"${res.get('std_M', 'N/A'):.1f}M")
            rc3.metric('Skewness', f"{res.get('skewness', 'N/A')}")
            rc4.metric('Max Underprediction', f"${res.get('max_underprediction_M', 'N/A'):.0f}M")
    else:
        st.info('Run notebook 06 and 07 Export cells to load model results.')

    st.markdown('---')
    st.subheader('Figures')
    fig_tab1, fig_tab2, fig_tab3 = st.tabs(['Feature Importance', 'Actual vs Predicted', 'Residuals'])

    with fig_tab1:
        fi_data = load_json('feature_importance_regression.json')
        if fi_data:
            fi_df = pd.DataFrame(fi_data).head(15).sort_values('importance')
            fig = px.bar(fi_df, x='importance', y='feature', orientation='h',
                         title='Feature Importance — Random Forest Regressor',
                         color='importance', color_continuous_scale='Purples')
            st.plotly_chart(fig, use_container_width=True)
        else:
            img = load_image('feature_importance_horizontal.png')
            if img: st.image(img, use_container_width=True)
            else: st.info('Run notebook 07 Export cell.')
        st.markdown('`budget` accounts for **63.6%** of feature importance.')

    with fig_tab2:
        img = load_image('actual_vs_predicted_rf.png')
        if img: st.image(img, use_container_width=True)
        else: st.info('Run notebook 07 Export cell.')

    with fig_tab3:
        img = load_image('residual_analysis.png')
        if img: st.image(img, use_container_width=True)
        else: st.info('Run notebook 07 Export cell.')
        st.markdown('Residual skew = 4.84 — extreme blockbusters are systematically underpredicted.')

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ML Classification
# ══════════════════════════════════════════════════════════════════════════════
elif page == '🏷️ ML — Classification':
    st.title('Machine Learning — Hit / Flop Classification')
    st.markdown('Binary classification: does a film earn above or below the dataset median revenue?')

    model_data = load_json('model_results.json')

    if model_data and 'classification' in model_data:
        clf = model_data['classification']
        threshold = clf.get('median_threshold_USD', 30_000_000)
        lr_c = clf.get('logistic_regression', {})
        rf_c = clf.get('random_forest', {})

        st.info(f'**Threshold:** Revenue > ${threshold/1e6:.1f}M = Hit (1), else Flop (0). Balanced dataset: 50% hits / 50% flops.')

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('**Logistic Regression**')
            st.metric('Accuracy', f"{lr_c.get('Accuracy', 'N/A')}")
            st.metric('F1 Score', f"{lr_c.get('F1', 'N/A')}")
            st.metric('AUC-ROC',  f"{lr_c.get('AUC-ROC', 'N/A')}")
            if 'cv_auc_mean' in lr_c:
                st.metric('CV AUC', f"{lr_c['cv_auc_mean']:.4f} ± {lr_c['cv_auc_std']:.4f}")
        with c2:
            st.markdown('**Random Forest Classifier**')
            st.metric('Accuracy', f"{rf_c.get('Accuracy', 'N/A')}")
            st.metric('F1 Score', f"{rf_c.get('F1', 'N/A')}")
            st.metric('AUC-ROC',  f"{rf_c.get('AUC-ROC', 'N/A')}")
            if 'cv_auc_mean' in rf_c:
                st.metric('CV AUC', f"{rf_c['cv_auc_mean']:.4f} ± {rf_c['cv_auc_std']:.4f}")
    else:
        st.info('Run notebook 08 Export cell to load classification results.')

    st.markdown('---')
    st.subheader('Figures')
    clf_tab1, clf_tab2, clf_tab3, clf_tab4 = st.tabs([
        'ROC Curve', 'Confusion Matrices', 'Metrics Comparison', 'Feature Importance'
    ])

    with clf_tab1:
        img = load_image('roc_curve.png')
        if img: st.image(img, use_container_width=True)
        else: st.info('Run notebook 08 Export cell.')
        st.markdown('Logistic Regression AUC = **0.860** — strong separation ability.')

    with clf_tab2:
        img = load_image('confusion_matrices.png')
        if img: st.image(img, use_container_width=True)
        else: st.info('Run notebook 08 Export cell.')

    with clf_tab3:
        img = load_image('classification_comparison.png')
        if img: st.image(img, use_container_width=True)
        else: st.info('Run notebook 08 Export cell.')

    with clf_tab4:
        fi_clf = load_json('feature_importance_classifier.json')
        if fi_clf:
            fi_df = pd.DataFrame(fi_clf).head(15).sort_values('importance')
            fig = px.bar(fi_df, x='importance', y='feature', orientation='h',
                         title='Feature Importance — Random Forest Classifier',
                         color='importance', color_continuous_scale='Purples')
            st.plotly_chart(fig, use_container_width=True)
        else:
            img = load_image('feature_importance_classifier.png')
            if img: st.image(img, use_container_width=True)
            else: st.info('Run notebook 08 Export cell.')
        st.markdown('`budget` = **47.2%** importance — same dominant feature as in regression.')

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Film Predictor
# ══════════════════════════════════════════════════════════════════════════════
elif page == '🎬 Film Predictor':
    st.title('Film Predictor')
    st.markdown('Enter a film\'s pre-release information to predict **revenue** and **hit/flop probability**.')

    model_pkg = load_model('random_forest_regressor.pkl')
    clf_pkg   = load_model('logistic_regression.pkl')

    if model_pkg is None or clf_pkg is None:
        st.warning(
            'Trained models not found in `Output/models/`. '
            'Run notebooks 06 and 08 (including the Export cells) first.'
        )
        st.stop()

    reg_model   = model_pkg['model']
    reg_features = model_pkg['features']
    clf_model    = clf_pkg['model']
    clf_scaler   = clf_pkg['scaler']
    clf_features = clf_pkg['features']
    threshold    = clf_pkg.get('threshold_USD', 29_918_744)

    known_genres = sorted({f.replace('primary_genre_', '') for f in reg_features if f.startswith('primary_genre_')})
    known_decades = sorted({
        int(f.replace('decade_', '').rstrip('s'))
        for f in reg_features if f.startswith('decade_')
    })

    st.markdown('---')
    col_in, col_out = st.columns([1, 1])

    with col_in:
        st.subheader('Film Details')
        budget = st.number_input('Production Budget (USD)', min_value=1, max_value=500_000_000,
                                  value=30_000_000, step=1_000_000, format='%d')
        runtime = st.slider('Runtime (minutes)', 60, 240, 110)
        release_year = st.number_input('Release Year', min_value=1960, max_value=2030, value=2024)
        genre = st.selectbox('Primary Genre', ['(none)'] + known_genres)
        decade_val = (release_year // 10) * 10

        st.caption(f'Decade auto-detected: **{decade_val}s**')
        predict_btn = st.button('Predict', type='primary', use_container_width=True)

    with col_out:
        st.subheader('Prediction')

        if predict_btn:
            row = {f: 0 for f in reg_features}
            row['budget']       = budget
            row['runtime']      = runtime
            row['release_year'] = release_year

            genre_col  = f'primary_genre_{genre}'
            decade_col = f'decade_{decade_val}s'
            if genre_col in row:
                row[genre_col] = 1
            if decade_col in row:
                row[decade_col] = 1

            X_reg = pd.DataFrame([row])[reg_features]
            revenue_pred = reg_model.predict(X_reg)[0]

            X_clf_raw = pd.DataFrame([{f: 0 for f in clf_features}])
            X_clf_raw['budget']       = budget
            X_clf_raw['runtime']      = runtime
            X_clf_raw['release_year'] = release_year
            if genre_col  in clf_features: X_clf_raw[genre_col]  = 1
            if decade_col in clf_features: X_clf_raw[decade_col] = 1
            X_clf = X_clf_raw[clf_features]
            X_clf_scaled = clf_scaler.transform(X_clf)
            hit_prob = clf_model.predict_proba(X_clf_scaled)[0][1]
            is_hit   = revenue_pred > threshold

            rev_m = revenue_pred / 1e6
            roi   = (revenue_pred - budget) / budget * 100

            st.metric('Predicted Revenue', f'${rev_m:,.1f}M')
            st.metric('ROI Estimate', f'{roi:+.0f}%')
            st.metric('Hit Probability', f'{hit_prob*100:.1f}%')

            if is_hit:
                st.success(f'**HIT** — predicted revenue ${rev_m:,.1f}M exceeds the ${threshold/1e6:.0f}M threshold.')
            else:
                st.error(f'**FLOP** — predicted revenue ${rev_m:,.1f}M is below the ${threshold/1e6:.0f}M threshold.')

            st.markdown('---')
            st.caption(
                f'Model: Random Forest Regressor (R²=0.542) for revenue · '
                f'Logistic Regression (AUC=0.860) for hit probability · '
                f'Threshold: ${threshold/1e6:.1f}M (dataset median)'
            )
        else:
            st.info('Fill in the film details and click **Predict**.')

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Dataset
# ══════════════════════════════════════════════════════════════════════════════
elif page == '📁 Dataset':
    st.title('Dataset — movies_merged.csv')
    st.markdown('Cleaned and merged dataset used in all analyses. 5,381 films · 20 columns.')

    df = load_df()

    if df is None:
        st.error(
            '`Data/processed/movies_merged.csv` not found. '
            'Run notebook `02_cleaning_and_merge.ipynb` to generate it.'
        )
        st.stop()

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.markdown('---')
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Rows', f'{len(df):,}')
    c2.metric('Columns', df.shape[1])
    c3.metric('Genres', df['primary_genre'].nunique() if 'primary_genre' in df.columns else '—')
    c4.metric('Year Range', f"{int(df['release_year'].min())}–{int(df['release_year'].max())}" if 'release_year' in df.columns else '—')
    c5.metric('Avg IMDb', f"{df['imdb_rating'].mean():.2f}" if 'imdb_rating' in df.columns else '—')

    st.markdown('---')

    # ── Filters ───────────────────────────────────────────────────────────────
    st.subheader('Filter')
    f1, f2, f3 = st.columns(3)

    with f1:
        genres = ['All'] + sorted(df['primary_genre'].dropna().unique().tolist()) if 'primary_genre' in df.columns else ['All']
        selected_genre = st.selectbox('Genre', genres)

    with f2:
        tiers = ['All'] + ['Low', 'Mid', 'High', 'Blockbuster']
        selected_tier = st.selectbox('Budget Tier', tiers)

    with f3:
        if 'release_year' in df.columns:
            year_min = int(df['release_year'].min())
            year_max = int(df['release_year'].max())
            year_range = st.slider('Release Year', year_min, year_max, (year_min, year_max))
        else:
            year_range = None

    # ── Search ────────────────────────────────────────────────────────────────
    search = st.text_input('Search by title', placeholder='e.g. Inception')

    # ── Apply filters ─────────────────────────────────────────────────────────
    filtered = df.copy()
    if selected_genre != 'All' and 'primary_genre' in filtered.columns:
        filtered = filtered[filtered['primary_genre'] == selected_genre]
    if selected_tier != 'All' and 'budget_tier' in filtered.columns:
        filtered = filtered[filtered['budget_tier'] == selected_tier]
    if year_range and 'release_year' in filtered.columns:
        filtered = filtered[filtered['release_year'].between(year_range[0], year_range[1])]
    if search.strip():
        filtered = filtered[filtered['title'].str.contains(search.strip(), case=False, na=False)]

    st.caption(f'Showing **{len(filtered):,}** of {len(df):,} films')

    # ── Display columns (readable subset first) ────────────────────────────────
    display_cols = ['title', 'release_year', 'primary_genre', 'budget_tier',
                    'budget', 'revenue', 'roi', 'imdb_rating', 'tomatometer_rating',
                    'audience_rating', 'runtime', 'decade']
    show_cols = [c for c in display_cols if c in filtered.columns]
    remaining = [c for c in filtered.columns if c not in show_cols]
    col_order = show_cols + remaining

    st.dataframe(
        filtered[col_order].reset_index(drop=True),
        use_container_width=True,
        height=480,
        column_config={
            'budget':             st.column_config.NumberColumn('Budget (USD)', format='$%d'),
            'revenue':            st.column_config.NumberColumn('Revenue (USD)', format='$%d'),
            'roi':                st.column_config.NumberColumn('ROI (%)', format='%.1f'),
            'imdb_rating':        st.column_config.NumberColumn('IMDb', format='%.1f'),
            'tomatometer_rating': st.column_config.NumberColumn('Tomatometer', format='%.0f'),
            'audience_rating':    st.column_config.NumberColumn('Audience', format='%.0f'),
            'runtime':            st.column_config.NumberColumn('Runtime (min)', format='%d'),
        },
    )

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown('---')
    dl1, dl2 = st.columns(2)

    with dl1:
        csv_bytes = filtered[col_order].to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f'Download filtered dataset ({len(filtered):,} rows) — CSV',
            data=csv_bytes,
            file_name='movies_filtered.csv',
            mime='text/csv',
            use_container_width=True,
        )

    with dl2:
        full_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f'Download full dataset ({len(df):,} rows) — CSV',
            data=full_csv,
            file_name='movies_merged.csv',
            mime='text/csv',
            use_container_width=True,
        )
