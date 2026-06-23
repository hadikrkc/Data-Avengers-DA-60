import ast

import pandas as pd


def normalize_title(series: pd.Series) -> pd.Series:
    """Standardize title strings for safer matching across datasets."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert a column to numeric values, coercing invalid values to NaN."""
    return pd.to_numeric(series, errors="coerce")


MIN_FINANCIAL_VALUE = 10_000  # Values below this are data entry errors (e.g. budget stored as $1 instead of $1M)

def filter_positive_budget_revenue(df: pd.DataFrame, budget_col: str, revenue_col: str) -> pd.DataFrame:
    """Keep rows with usable budget and revenue values.

    Threshold is $10,000: values below this are almost certainly data entry errors
    (budget/revenue recorded in thousands or as placeholder $1 values), not real
    productions with sub-$10K financials.
    """
    cleaned = df.copy()
    cleaned[budget_col] = coerce_numeric(cleaned[budget_col])
    cleaned[revenue_col] = coerce_numeric(cleaned[revenue_col])
    return cleaned[
        (cleaned[budget_col] >= MIN_FINANCIAL_VALUE) &
        (cleaned[revenue_col] >= MIN_FINANCIAL_VALUE)
    ]


def extract_year(series: pd.Series) -> pd.Series:
    """Parse a date string column into integer release years."""
    return pd.to_datetime(series, errors="coerce").dt.year


def extract_primary_genre(series: pd.Series) -> pd.Series:
    """Extract the first genre name from a JSON-like genre list string."""
    def _parse(val):
        try:
            genres = ast.literal_eval(str(val))
            if isinstance(genres, list) and genres:
                return genres[0].get("name")
        except (ValueError, SyntaxError):
            pass
        return None
    return series.apply(_parse)


def add_roi_column(df: pd.DataFrame, budget_col: str, revenue_col: str) -> pd.DataFrame:
    """Add roi = (revenue - budget) / budget * 100 column."""
    result = df.copy()
    result["roi"] = (result[revenue_col] - result[budget_col]) / result[budget_col] * 100
    return result


def add_budget_tier(df: pd.DataFrame, budget_col: str) -> pd.DataFrame:
    """Assign Low / Mid / High / Blockbuster labels based on budget quartiles."""
    result = df.copy()
    result["budget_tier"] = pd.qcut(
        result[budget_col],
        q=4,
        labels=["Low", "Mid", "High", "Blockbuster"],
        duplicates="drop",
    )
    return result


def add_decade(df: pd.DataFrame, year_col: str) -> pd.DataFrame:
    """Derive a decade label (e.g. '2000s') from a release year column."""
    result = df.copy()
    years = pd.to_numeric(result[year_col], errors="coerce")
    result["decade"] = years.apply(
        lambda y: f"{int(y // 10 * 10)}s" if pd.notna(y) else None
    )
    return result
