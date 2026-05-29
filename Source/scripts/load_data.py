from pathlib import Path

import pandas as pd

from .helpers import RAW_DATA_DIR


DATASET_PATHS = {
    "the_movies_metadata": RAW_DATA_DIR / "The Movies Dataset" / "movies_metadata.csv",
    "the_movies_credits": RAW_DATA_DIR / "The Movies Dataset" / "credits.csv",
    "tmdb_movies": RAW_DATA_DIR / "TMDB 5000 Movies Dataset" / "tmdb_5000_movies.csv",
    "tmdb_credits": RAW_DATA_DIR / "TMDB 5000 Movies Dataset" / "tmdb_5000_credits.csv",
    "rt_movies": RAW_DATA_DIR / "Rotten Tomatoes movies and critic reviews dataset" / "rotten_tomatoes_movies.csv",
    "rt_reviews": RAW_DATA_DIR / "Rotten Tomatoes movies and critic reviews dataset" / "rotten_tomatoes_critic_reviews.csv",
    "imdb_basics": RAW_DATA_DIR / "IMDb Dataset" / "title.basics.tsv",
    "imdb_ratings": RAW_DATA_DIR / "IMDb Dataset" / "title.ratings.tsv",
}


def load_table(path: Path, **kwargs) -> pd.DataFrame:
    """Load a CSV or TSV file into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False, **kwargs)

    return pd.read_csv(path, low_memory=False, **kwargs)


def load_named_dataset(name: str, **kwargs) -> pd.DataFrame:
    """Load one of the predefined raw datasets by name."""
    if name not in DATASET_PATHS:
        available = ", ".join(sorted(DATASET_PATHS))
        raise KeyError(f"Unknown dataset '{name}'. Available datasets: {available}")

    return load_table(DATASET_PATHS[name], **kwargs)
