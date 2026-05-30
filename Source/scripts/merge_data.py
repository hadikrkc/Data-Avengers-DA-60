import pandas as pd

from .clean_data import (
    add_budget_tier,
    add_decade,
    add_roi_column,
    extract_primary_genre,
    extract_year,
    filter_positive_budget_revenue,
    normalize_title,
)
from .helpers import ensure_processed_dir
from .load_data import load_named_dataset


def merge_on_title(left: pd.DataFrame, right: pd.DataFrame, left_title: str, right_title: str) -> pd.DataFrame:
    """Merge two DataFrames on standardized title columns."""
    return left.merge(right, left_on=left_title, right_on=right_title, how="left")


def build_master_dataset() -> pd.DataFrame:
    """Build the cleaned, merged master dataset and save to Data/processed/movies_merged.csv."""

    # Step 1: movies_metadata — primary source for budget, revenue, imdb_id
    print("Loading movies_metadata...")
    meta = load_named_dataset("the_movies_metadata")
    meta = filter_positive_budget_revenue(meta, "budget", "revenue")
    print(f"  After budget/revenue filter: {len(meta)} rows")

    meta["title_clean"] = normalize_title(meta["title"])
    meta["release_year"] = extract_year(meta["release_date"])
    meta["primary_genre"] = extract_primary_genre(meta["genres"])

    keep = [c for c in [
        "id", "imdb_id", "title", "title_clean", "budget", "revenue",
        "runtime", "release_year", "primary_genre", "vote_average", "vote_count", "popularity",
    ] if c in meta.columns]
    df = meta[keep].copy()

    # Step 2: TMDB — supplement with tmdb_popularity
    print("Loading tmdb_movies...")
    tmdb = load_named_dataset("tmdb_movies")
    tmdb["title_clean"] = normalize_title(tmdb["title"])
    if "popularity" in tmdb.columns:
        tmdb_slim = (
            tmdb[["title_clean", "popularity"]]
            .rename(columns={"popularity": "tmdb_popularity"})
            .drop_duplicates("title_clean")
        )
        df = df.merge(tmdb_slim, on="title_clean", how="left")
    print(f"  After TMDB merge: {len(df)} rows")

    # Step 3: IMDb ratings — join on imdb_id (more reliable than title matching)
    print("Loading IMDb ratings...")
    imdb = load_named_dataset("imdb_ratings")
    imdb = imdb.rename(columns={
        "tconst": "imdb_id",
        "averageRating": "imdb_rating",
        "numVotes": "imdb_votes",
    })
    df = df.merge(imdb[["imdb_id", "imdb_rating", "imdb_votes"]], on="imdb_id", how="left")
    print(f"  After IMDb merge: {len(df)} rows | imdb_rating filled: {df['imdb_rating'].notna().sum()}")

    # Step 4: Rotten Tomatoes — join on normalized title
    print("Loading Rotten Tomatoes...")
    rt = load_named_dataset("rt_movies")
    title_col = "movie_title" if "movie_title" in rt.columns else "title"
    rt["title_clean"] = normalize_title(rt[title_col])
    rt_cols = [c for c in ["title_clean", "tomatometer_rating", "audience_rating"] if c in rt.columns]
    rt_slim = rt[rt_cols].drop_duplicates("title_clean")
    df = df.merge(rt_slim, on="title_clean", how="left")
    if "tomatometer_rating" in df.columns:
        print(f"  After RT merge: {len(df)} rows | tomatometer filled: {df['tomatometer_rating'].notna().sum()}")
    else:
        print(f"  After RT merge: {len(df)} rows")

    # Step 5: Derived columns
    df = add_roi_column(df, "budget", "revenue")
    df = add_budget_tier(df, "budget")
    df = add_decade(df, "release_year")

    # Step 6: Save
    out_path = ensure_processed_dir() / "movies_merged.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")

    return df
