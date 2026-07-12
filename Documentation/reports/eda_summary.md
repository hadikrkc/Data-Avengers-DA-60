# EDA Summary — Budget vs Success: Movie Data Analysis

**Dataset:** `Data/processed/movies_merged.csv` — 5,280 films · 20 columns  
**Notebook:** `Source/notebooks/03_eda.ipynb`  
**Generated:** Session 3 · Updated after data quality cleaning (Phase 8/9)

---

## Key Findings

### 1. Budget → Revenue (r = 0.65)
The strongest signal in the dataset. Log-log Pearson r = **0.650** (n = 5,280, p < .001); r² ≈ 0.42 means budget explains ~42% of revenue variance. Higher budget reliably predicts higher revenue, but scatter is wide — especially at the top end. Budget is a necessary but not sufficient condition for box office success.

> **Data quality note:** Previous value was r = 0.704 on the uncleaned dataset (5,381 films including
> entries with budget as low as $1). After removing 101 records with budget or revenue < $10,000
> (data entry errors), r = 0.650. The relationship remains strong and the conclusion is unchanged.

### 2. Budget → IMDb Rating (r = -0.094)
Near-zero correlation (n = 5,278, p < .001). Statistically significant only because of the large sample size — budget explains less than 1% of IMDb score variance. Spending more does not produce better audience ratings. Quality cannot be bought.

### 3. Budget → Tomatometer (r = -0.221)
Statistically significant (n = 4,326, p < .001) and moderately negative. Higher-budget films score lower with critics, likely because large studios favor commercial safety over artistic risk. This signal is slightly stronger after removing data entry error records.

### 4. ROI by Budget Tier

| Tier | Median ROI |
|---|---|
| Low | **189%** |
| Mid | 80% |
| High | 70% |
| Blockbuster | 116% |

Low-budget films offer the best median return. The Mid–High range is the riskiest: large spend without blockbuster draw. Blockbuster tier recovers via franchise/IP reliability.

### 5. IMDb Rating by Budget Tier

| Tier | Avg IMDb |
|---|---|
| Low | **6.67** |
| Mid | 6.52 |
| High | 6.38 |
| Blockbuster | 6.46 |

Consistent slight decline with budget (~0.3 points across all tiers). May partly reflect selection bias — only notable low-budget films enter the dataset.

### 6. Top ROI Films — All Low Budget
Every film in the Top 20 ROI list is in the Low tier. Notable examples:
- **Blair Witch Project**: ~$60K budget → $248M revenue
- **Paranormal Activity**: minimal budget → $193M revenue
- Horror and Drama dominate; lean production enables massive ROI.

### 7. Biggest Flops — All Blockbuster Tier
Every film in the Bottom 20 ROI list is in the Blockbuster tier. Notable examples:
- **Foodfight!**: $65M budget → $0.1M revenue (ROI: -100%)
- **The Adventures of Pluto Nash**: $100M budget → $7.1M revenue (ROI: -93%)
- Action and Drama dominate; high spend with poor creative execution is catastrophic.

### 8. Budget by Genre

| Genre | Median Budget |
|---|---|
| Animation | $59.5M |
| Adventure | $40.0M |
| Family | $35.0M |
| Action | $28.0M |
| Science Fiction | $27.5M |

Animation leads due to CGI costs. Horror and Documentary are absent from the top 10 — they operate in the Low tier and consequently deliver the highest ROI.

### 9. Decade Trend

| Decade | Median Budget | Median Revenue |
|---|---|---|
| 1910s | $0.2M | $8.0M |
| 1950s | $2.0M | $9.0M |
| 1980s | $11.0M | $21.6M |
| 1990s | $22.0M | $25.9M |
| 2000s | $25.0M | $38.6M |
| 2010s | $19.8M | $36.9M |

Both metrics grew steadily across the century. Revenue consistently exceeded budget in every decade — the industry is profitable on a median basis. The 2010s budget dip with stable revenue may reflect streaming competition narrowing theatrical ROI.

---

## Answer to the Research Question

**Does budget drive success?**

- **Financially (revenue):** Yes — strong correlation (r = 0.650). Budget is the single strongest predictor of box office revenue.
- **Critically (IMDb / Tomatometer):** No — near-zero or negative. More money does not produce better films by audience or critic standards; the negative Tomatometer relationship (r = -0.221) is more pronounced than previously measured.
- **Return on investment:** Counterintuitively, low-budget films deliver the best median ROI (189% vs 70–116% for larger tiers).

The data suggests budget buys *scale of audience* but not *quality of reception*. A blockbuster budget virtually guarantees a large release but does not predict whether audiences or critics will love it.

---

## Next Step
→ `04_final_analysis.ipynb` — visualization and final insight synthesis
