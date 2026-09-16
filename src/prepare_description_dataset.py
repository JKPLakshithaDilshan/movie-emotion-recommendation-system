from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd


RAW_DATA_PATH = Path(
    "data/raw/Movies_Reviews_modified_version1.csv"
)

FULL_OUTPUT_PATH = Path(
    "data/processed/description_level_dataset.csv"
)

MODELLING_OUTPUT_PATH = Path(
    "data/processed/modelling_dataset_7class.csv"
)

RANDOM_STATE = 42
EXCLUDED_MODEL_CLASS = "surprise"


def parse_genres(value: str) -> list[str]:
    """Convert a stored genre-list string into a Python list."""
    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [
                str(genre).strip()
                for genre in parsed
                if str(genre).strip()
            ]

    except (ValueError, SyntaxError, TypeError):
        pass

    return []


def join_unique_values(series: pd.Series) -> str:
    """Join unique non-empty values in deterministic order."""
    values = sorted(
        {
            str(value).strip()
            for value in series
            if pd.notna(value) and str(value).strip()
        }
    )

    return " | ".join(values)


def validate_columns(
    df: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """Verify that all required columns exist in the raw dataset."""
    missing_columns = sorted(
        set(required_columns) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Raw dataset is missing required columns: "
            f"{missing_columns}"
        )


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Loaded raw dataset: {df.shape[0]:,} rows")

    required_columns = [
        "Ratings",
        "Reviews",
        "movie_name",
        "Resenhas",
        "genres",
        "Description",
        "emotion",
    ]

    validate_columns(df, required_columns)

    # Remove the exported row-number column because it has no
    # predictive meaning and would introduce noise or leakage.
    df = df.drop(
        columns=["Unnamed: 0"],
        errors="ignore",
    )

    text_columns = [
        "Reviews",
        "movie_name",
        "Resenhas",
        "genres",
        "Description",
        "emotion",
    ]

    for column in text_columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    # Convert empty or whitespace-only strings into real missing values.
    df = df.replace(
        r"^\s*$",
        pd.NA,
        regex=True,
    )

    missing_before = df.isna().sum()

    print("\nMissing values before required-field handling:")
    print(missing_before)

    # Records without the target or essential modelling fields
    # cannot be used safely.
    df = df.dropna(
        subset=[
            "Description",
            "emotion",
            "movie_name",
            "genres",
            "Ratings",
        ]
    ).copy()

    # Ratings outside the documented 1-10 scale are invalid.
    invalid_rating_count = (
        ~df["Ratings"].between(1, 10)
    ).sum()

    print(
        "\nRatings outside the valid 1-10 range:",
        f"{invalid_rating_count:,}",
    )

    if invalid_rating_count:
        df = df[
            df["Ratings"].between(1, 10)
        ].copy()

    duplicate_count = df.duplicated().sum()

    print(
        f"Exact duplicate records: {duplicate_count:,}"
    )

    if duplicate_count:
        df = df.drop_duplicates().copy()

    # An identical description must not have conflicting
    # emotion labels.
    label_counts_per_description = (
        df.groupby("Description")["emotion"].nunique()
    )

    conflicting_descriptions = (
        label_counts_per_description[
            label_counts_per_description > 1
        ]
    )

    if not conflicting_descriptions.empty:
        raise ValueError(
            "Some descriptions have conflicting emotion labels."
        )

    # Create one independent record per unique description.
    description_df = (
        df.groupby(
            ["Description", "emotion"],
            as_index=False,
            sort=True,
        )
        .agg(
            movie_name=(
                "movie_name",
                "first",
            ),
            all_movie_names=(
                "movie_name",
                join_unique_values,
            ),
            movie_name_count=(
                "movie_name",
                "nunique",
            ),
            genres=(
                "genres",
                "first",
            ),
            mean_rating=(
                "Ratings",
                "mean",
            ),
            median_rating=(
                "Ratings",
                "median",
            ),
            minimum_rating=(
                "Ratings",
                "min",
            ),
            maximum_rating=(
                "Ratings",
                "max",
            ),
            review_count=(
                "Reviews",
                "size",
            ),
            unique_review_count=(
                "Reviews",
                "nunique",
            ),
        )
        .reset_index(drop=True)
    )

    # Parse the genre strings and create categorical features.
    description_df["genre_list"] = (
        description_df["genres"].apply(parse_genres)
    )

    description_df["genre_count"] = (
        description_df["genre_list"].str.len()
    )

    description_df["genres_text"] = (
        description_df["genre_list"].apply(
            lambda values: " ".join(
                genre.lower().replace(" ", "_")
                for genre in values
            )
        )
    )

    # Empty genre lists represent unavailable categorical
    # information. Preserve the records using an explicit category.
    description_df["genres_text"] = (
        description_df["genres_text"]
        .replace("", "unknown_genre")
        .fillna("unknown_genre")
    )

    # Engineer description-length features.
    description_df["description_word_count"] = (
        description_df["Description"]
        .str.split()
        .str.len()
    )

    description_df["description_character_count"] = (
        description_df["Description"].str.len()
    )

    # Reduce the effect of extreme review-count values.
    description_df["log_review_count"] = np.log1p(
        description_df["review_count"]
    )

    description_df[
        "log_unique_review_count"
    ] = np.log1p(
        description_df["unique_review_count"]
    )

    # Store the parsed list in CSV-compatible form.
    description_df["genre_list"] = (
        description_df["genre_list"].apply(str)
    )

    description_df = (
        description_df.sort_values(
            [
                "emotion",
                "movie_name",
                "Description",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    # Validate the processed dataset before writing files.
    if description_df[
        "Description"
    ].duplicated().any():
        raise ValueError(
            "Description-level dataset still contains "
            "duplicate descriptions."
        )

    remaining_missing = description_df.isna().sum()

    if remaining_missing.sum() > 0:
        raise ValueError(
            "Processed dataset contains missing values:\n"
            f"{remaining_missing[remaining_missing > 0]}"
        )

    if not description_df[
        "mean_rating"
    ].between(1, 10).all():
        raise ValueError(
            "Processed mean ratings are outside the "
            "valid 1-10 range."
        )

    FULL_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    description_df.to_csv(
        FULL_OUTPUT_PATH,
        index=False,
    )

    # Surprise has only three independent description records.
    # It remains in the full EDA dataset but is excluded from
    # classification because reliable stratified validation is
    # impossible with only three examples.
    modelling_df = description_df[
        description_df["emotion"]
        != EXCLUDED_MODEL_CLASS
    ].copy()

    modelling_df.to_csv(
        MODELLING_OUTPUT_PATH,
        index=False,
    )

    print("\nFull description-level dataset:")
    print(f"Rows: {len(description_df):,}")
    print(
        description_df["emotion"].value_counts()
    )

    print("\nSeven-class modelling dataset:")
    print(f"Rows: {len(modelling_df):,}")
    print(
        modelling_df["emotion"].value_counts()
    )

    print(
        "\nUnknown-genre records:",
        int(
            description_df["genres_text"]
            .eq("unknown_genre")
            .sum()
        ),
    )

    print(
        "Remaining missing values:",
        int(description_df.isna().sum().sum()),
    )

    print("\nSaved:")
    print(FULL_OUTPUT_PATH)
    print(MODELLING_OUTPUT_PATH)

    print(
        f"\nRandom-state standard: {RANDOM_STATE}"
    )
    print(
        "Dataset preparation completed successfully."
    )


if __name__ == "__main__":
    main()