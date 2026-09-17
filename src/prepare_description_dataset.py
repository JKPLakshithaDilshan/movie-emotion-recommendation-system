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

REQUIRED_COLUMNS = [
    "Ratings",
    "Reviews",
    "movie_name",
    "Resenhas",
    "genres",
    "Description",
    "emotion",
]


def parse_genres(value: object) -> list[str]:
    """
    Convert a stored genre-list string into a cleaned Python list.

    Example:
        "['Drama', 'Romance']" -> ['Drama', 'Romance']
        "[]" -> []
    """
    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(str(value))

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
    dataframe: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """Verify that all required columns exist."""
    missing_columns = sorted(
        set(required_columns) - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Raw dataset is missing required columns: "
            f"{missing_columns}"
        )


def validate_processed_dataset(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Run final data-quality checks before saving."""
    duplicate_descriptions = int(
        dataframe["Description"].duplicated().sum()
    )

    if duplicate_descriptions > 0:
        raise ValueError(
            f"{dataset_name} contains "
            f"{duplicate_descriptions} duplicated descriptions."
        )

    remaining_missing = dataframe.isna().sum()
    remaining_missing = remaining_missing[
        remaining_missing > 0
    ]

    if not remaining_missing.empty:
        raise ValueError(
            f"{dataset_name} contains missing values:\n"
            f"{remaining_missing}"
        )

    if not dataframe["mean_rating"].between(1, 10).all():
        raise ValueError(
            f"{dataset_name} contains mean ratings "
            "outside the valid 1-10 range."
        )

    empty_genres_text = (
        dataframe["genres_text"]
        .astype("string")
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_genres_text > 0:
        raise ValueError(
            f"{dataset_name} contains "
            f"{empty_genres_text} empty genres_text values."
        )


def main() -> None:
    """Prepare description-level datasets for EDA and modelling."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    print(
        f"Loaded raw dataset: "
        f"{df.shape[0]:,} rows and {df.shape[1]} columns"
    )

    validate_columns(df, REQUIRED_COLUMNS)

    # Remove the exported row-number column. It has no predictive
    # meaning and could introduce unnecessary noise.
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

    # Standardize the target labels.
    df["emotion"] = df["emotion"].str.lower()

    # Convert whitespace-only strings into real missing values.
    df = df.replace(
        r"^\s*$",
        pd.NA,
        regex=True,
    )

    # Convert ratings safely to numeric values.
    df["Ratings"] = pd.to_numeric(
        df["Ratings"],
        errors="coerce",
    )

    print("\nMissing values before required-field handling:")
    print(df.isna().sum())

    rows_before_required_handling = len(df)

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

    dropped_required_rows = (
        rows_before_required_handling - len(df)
    )

    print(
        "\nRows removed because essential values were missing:",
        f"{dropped_required_rows:,}",
    )

    # Ratings outside the documented 1-10 range are invalid.
    valid_rating_mask = df["Ratings"].between(
        1,
        10,
        inclusive="both",
    )

    invalid_rating_count = int(
        (~valid_rating_mask).sum()
    )

    print(
        "Ratings outside the valid 1-10 range:",
        f"{invalid_rating_count:,}",
    )

    if invalid_rating_count > 0:
        df = df.loc[valid_rating_mask].copy()

    duplicate_count = int(df.duplicated().sum())

    print(
        "Exact duplicate records:",
        f"{duplicate_count:,}",
    )

    if duplicate_count > 0:
        df = df.drop_duplicates().copy()

    # The same description must not have multiple emotion labels.
    label_counts_per_description = (
        df.groupby(
            "Description",
            dropna=False,
        )["emotion"]
        .nunique()
    )

    conflicting_descriptions = (
        label_counts_per_description[
            label_counts_per_description > 1
        ]
    )

    print(
        "Descriptions with conflicting emotion labels:",
        f"{len(conflicting_descriptions):,}",
    )

    if not conflicting_descriptions.empty:
        raise ValueError(
            "Some descriptions have conflicting emotion labels."
        )

    # Create one independent record for each unique description.
    description_df = (
        df.groupby(
            [
                "Description",
                "emotion",
            ],
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

    # Parse genre-list strings.
    description_df["genre_list"] = (
        description_df["genres"].apply(parse_genres)
    )

    # Feature engineering: number of genres.
    description_df["genre_count"] = (
        description_df["genre_list"].str.len()
    )

    # Convert genre lists into text tokens suitable for vectorization.
    description_df["genres_text"] = (
        description_df["genre_list"].apply(
            lambda values: " ".join(
                genre.lower()
                .strip()
                .replace(" ", "_")
                for genre in values
            )
        )
    )

    # An empty genre list represents unavailable categorical
    # information. Preserve the record using an explicit category.
    description_df["genres_text"] = (
        description_df["genres_text"]
        .astype("string")
        .str.strip()
        .replace("", "unknown_genre")
        .fillna("unknown_genre")
    )

    # Feature engineering: description length.
    description_df["description_word_count"] = (
        description_df["Description"]
        .str.split()
        .str.len()
        .astype("int64")
    )

    description_df[
        "description_character_count"
    ] = (
        description_df["Description"]
        .str.len()
        .astype("int64")
    )

    # Feature engineering: rating spread.
    description_df["rating_range"] = (
        description_df["maximum_rating"]
        - description_df["minimum_rating"]
    )

    # Log-transform count features to reduce positive skew and
    # the influence of very large review-count observations.
    description_df["log_review_count"] = np.log1p(
        description_df["review_count"]
    )

    description_df[
        "log_unique_review_count"
    ] = np.log1p(
        description_df["unique_review_count"]
    )

    # Save the parsed genre list in a CSV-compatible format.
    description_df["genre_list"] = (
        description_df["genre_list"].apply(str)
    )

    # Use a deterministic ordering for reproducibility.
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

    validate_processed_dataset(
        description_df,
        "Full description-level dataset",
    )

    # The surprise class contains only three independent description
    # records. It is retained in the full dataset for EDA but excluded
    # from classification because reliable stratified validation is
    # not possible with only three observations.
    modelling_df = description_df.loc[
        description_df["emotion"]
        != EXCLUDED_MODEL_CLASS
    ].copy()

    modelling_df = modelling_df.reset_index(drop=True)

    validate_processed_dataset(
        modelling_df,
        "Seven-class modelling dataset",
    )

    FULL_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    description_df.to_csv(
        FULL_OUTPUT_PATH,
        index=False,
    )

    modelling_df.to_csv(
        MODELLING_OUTPUT_PATH,
        index=False,
    )

    print("\nFull description-level dataset:")
    print(f"Rows: {len(description_df):,}")
    print(f"Columns: {description_df.shape[1]}")
    print(
        description_df["emotion"]
        .value_counts()
        .to_string()
    )

    print("\nSeven-class modelling dataset:")
    print(f"Rows: {len(modelling_df):,}")
    print(f"Columns: {modelling_df.shape[1]}")
    print(
        modelling_df["emotion"]
        .value_counts()
        .to_string()
    )

    print(
        "\nUnknown-genre records in full dataset:",
        int(
            description_df["genres_text"]
            .eq("unknown_genre")
            .sum()
        ),
    )

    print(
        "Unknown-genre records in modelling dataset:",
        int(
            modelling_df["genres_text"]
            .eq("unknown_genre")
            .sum()
        ),
    )

    print(
        "Remaining missing values in full dataset:",
        int(description_df.isna().sum().sum()),
    )

    print(
        "Remaining missing values in modelling dataset:",
        int(modelling_df.isna().sum().sum()),
    )

    print(
        "Duplicated descriptions in modelling dataset:",
        int(
            modelling_df["Description"]
            .duplicated()
            .sum()
        ),
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