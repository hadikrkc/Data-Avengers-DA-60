# Data Overview

## Datasets Used

| Source | File | Total Rows | Usable Rows* | Join Key |
|---|---|---|---|---|
| The Movies Dataset (Kaggle) | movies_metadata.csv | 45,466 | 5,381 | imdb_id (primary key) |
| TMDB 5000 Movies Dataset | tmdb_5000_movies.csv | 4,803 | 3,229 | title (normalized) |
| IMDb Dataset | title.ratings.tsv | 1,666,284 | all | tconst = imdb_id |
| Rotten Tomatoes | rotten_tomatoes_movies.csv | 17,712 | all | movie_title (normalized) |

*Usable = budget > 0 AND revenue > 0

---

## Columns Selected Per Dataset

### movies_metadata.csv — Primary Source
| Column | Type | Notes |
|---|---|---|
| budget | numeric | Stored as string, coerced to float |
| revenue | numeric | Stored as string, coerced to float |
| title | string | Used for title matching |
| imdb_id | string | tt-prefixed ID, used to join IMDb ratings |
| genres | JSON string | Parsed to extract first genre name |
| release_date | string | Parsed to extract release_year (int) |
| runtime | numeric | Minutes |
| vote_average | numeric | TMDB community rating (fallback if IMDb unavailable) |
| vote_count | numeric | Number of TMDB votes |
| popularity | numeric | TMDB popularity score |

### tmdb_5000_movies.csv — Supplement
| Column | Type | Notes |
|---|---|---|
| popularity | numeric | Joined as `tmdb_popularity` via normalized title |

### title.ratings.tsv (IMDb) — Rating Source
| Column | Type | Notes |
|---|---|---|
| tconst | string | Matches imdb_id from movies_metadata (tt prefix) |
| averageRating | float | Renamed to `imdb_rating` |
| numVotes | int | Renamed to `imdb_votes` |

### rotten_tomatoes_movies.csv — Secondary Rating Source
| Column | Type | Notes |
|---|---|---|
| movie_title | string | Title column name (not `title`) — normalized for join |
| tomatometer_rating | float | Critic score — null rate 0.2% |
| audience_rating | float | Audience score — null rate 1.7% |

---

## Key Observations

- **Primary dataset size:** After filtering zero-budget/revenue rows, movies_metadata provides **5,381** usable films.
- **IMDb join reliability:** `imdb_id` in movies_metadata uses the same `tt` prefix format as IMDb's `tconst` — direct join is safe.
- **RT join:** Done via normalized title (`movie_title` column). Some mismatches expected due to punctuation or subtitle differences.
- **Rating coverage:** IMDb ratings are highly complete for films with a valid `imdb_id`. RT tomatometer is nearly complete (0.2% null).
- **Budget/revenue note:** movies_metadata stores `budget` and `revenue` as strings in some rows — `pd.to_numeric(..., errors='coerce')` is applied before filtering.

---

## Derived Columns Added in Cleaning Step

| Column | Formula | Purpose |
|---|---|---|
| title_clean | lowercase + strip + collapse whitespace | Safe cross-dataset matching |
| release_year | extracted from release_date | Numeric year for analysis and ML |
| primary_genre | first item from genres JSON list | Single genre label per film |
| roi | (revenue - budget) / budget * 100 | Return on investment |
| budget_tier | quartile cut: Low / Mid / High / Blockbuster | Budget group label |
| decade | release_year // 10 * 10, e.g. 2000s | Era grouping |
