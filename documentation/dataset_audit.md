# Dataset Audit and Modelling-Unit Decision

## 1. Dataset identity

**Dataset:** IMDb Movie Reviews – Genres, Descriptions and Emotions  
**Format:** CSV  
**Raw file:** `data/raw/Movies_Reviews_modified_version1.csv`  
**SHA-256:** `81ca8d80d2285b75e5f14cb0e7daefe532118ac0155583df8c6b9d1296bdc580`  
**Original size:** Approximately 129 MB  
**Original shape:** 46,173 rows and 8 columns

## 2. Raw columns

The raw dataset contains:

- `Unnamed: 0`
- `Ratings`
- `Reviews`
- `movie_name`
- `Resenhas`
- `genres`
- `Description`
- `emotion`

## 3. Initial data-quality findings

### Missing values

The raw dataset contained no literal null values in any column.

Blank-string testing also found no whitespace-only values.

However, after parsing the genre-list strings, 35 independent descriptions had empty genre lists represented by `[]`. These records were retained and assigned the explicit category `unknown_genre`.

### Artificial index column

`Unnamed: 0` contained 46,173 unique sequential values from 0 to 46,172.

It was removed because it was an exported row number and had no predictive meaning.

### Rating validity

All ratings were within the documented 1–10 range.

- Minimum: 1
- Maximum: 10
- Mean: approximately 5.98
- Median: 6

No records required removal because of invalid ratings.

## 4. Duplicate analysis

### Exact duplicates

The artificial index initially prevented duplicate detection because every index value was unique.

After removing `Unnamed: 0`, the audit found:

- 2,192 duplicate records
- 19,316 unique English reviews
- 26,857 repeated English-review rows
- 6,948 review texts repeated at least once

Exact duplicates were removed before aggregation.

### Conflicting review labels

The audit found 3,188 repeated review texts associated with more than one emotion label.

Some review texts appeared up to 72 times and were linked to as many as five emotion labels.

Therefore, English reviews were not selected as the primary emotion-classification input. Using them directly would introduce label ambiguity and could cause identical text to appear in training and test sets with different targets.

### Description consistency

The dataset contained 2,060 unique descriptions.

Every identical description had one consistent emotion label. This made the description the most defensible independent prediction unit.

### Movie-name ambiguity

The dataset contained 1,583 unique movie-name strings, but 273 names were linked with multiple emotion labels and 316 names were linked with multiple descriptions.

Names such as `Hero`, `Love`, and `Beauty and the Beast` represented multiple movies or versions. Therefore, `movie_name` was retained for display and recommendation output but not treated as a unique identifier or predictive feature.

## 5. Selected observation level

The modelling unit is one unique movie description.

The preparation process:

1. Removes the artificial index
2. Standardizes text fields
3. Converts blank strings into missing-value markers
4. Drops records missing essential fields, if any
5. Validates the 1–10 rating range
6. Removes exact duplicates
7. Confirms label consistency for identical descriptions
8. Aggregates repeated rows into one description-level record
9. Creates engineered features
10. Saves full EDA and modelling datasets

## 6. Description-level dataset

The full description-level dataset contains:

- 2,060 independent descriptions
- 19 columns
- 0 missing values
- 0 duplicated descriptions
- 35 records assigned `unknown_genre`

### Full class distribution

| Emotion | Count | Approximate percentage |
|---|---:|---:|
| Sadness | 817 | 39.66% |
| Joy | 400 | 19.42% |
| Anticipation | 249 | 12.09% |
| Optimism | 237 | 11.50% |
| Fear | 150 | 7.28% |
| Anger | 131 | 6.36% |
| Disgust | 73 | 3.54% |
| Surprise | 3 | 0.15% |

## 7. Surprise-class decision

Only three independent descriptions belong to `surprise`.

Three observations are insufficient for reliable stratified train-test splitting and five-fold cross-validation. Synthetic oversampling cannot create genuine linguistic diversity from only three independent examples.

Therefore:

- `surprise` remains in the full EDA dataset
- The limitation is reported transparently
- `surprise` is excluded from supervised model training
- The final supervised task uses seven emotion classes

This decision avoids reporting unreliable performance for an unsupported class.

## 8. Seven-class modelling dataset

The final modelling dataset contains:

- 2,057 independent descriptions
- Seven target classes
- 19 columns
- 0 missing values
- 0 duplicated descriptions

### Modelling class distribution

| Emotion | Count |
|---|---:|
| Sadness | 817 |
| Joy | 400 |
| Anticipation | 249 |
| Optimism | 237 |
| Fear | 150 |
| Anger | 131 |
| Disgust | 73 |

The class distribution remains imbalanced. Therefore, model evaluation must not rely only on accuracy.

Primary evaluation measures include:

- Macro F1-score
- Macro precision
- Macro recall
- Weighted F1-score
- Per-class recall
- Confusion matrix
- Stratified cross-validation

## 9. Selected model inputs

### Primary feature

- `Description`: transformed using TF-IDF

### Supporting categorical feature

- `genres_text`: encoded genre tokens
- Empty genre lists use `unknown_genre`

### Supporting numerical features

- `mean_rating`
- `median_rating`
- `minimum_rating`
- `maximum_rating`
- `review_count`
- `unique_review_count`
- `genre_count`
- `description_word_count`
- `description_character_count`
- `log_review_count`
- `log_unique_review_count`

### Non-predictive fields

The following fields are retained for identification, reporting, or recommendation output but are not direct identifiers for model learning:

- `movie_name`
- `all_movie_names`

## 10. Outlier policy

Valid long descriptions and popular movies are not automatically errors.

The project uses:

- Range validation for ratings
- IQR analysis for identifying extreme numerical values
- Log transformation for highly skewed review counts
- Training-only outlier boundaries where removal or capping affects modelling
- Before-and-after visualizations in every student notebook

This approach prevents valid popular movies from being removed without evidence.

## 11. Leakage prevention

To avoid leakage:

- Duplicate descriptions are removed before splitting
- Every description appears only once
- Train-test splitting uses stratification
- TF-IDF is fitted only on training data
- Encoders are fitted only on training data
- Scalers are fitted only on training data
- Feature selectors are fitted only on training data
- TruncatedSVD is fitted only on training data
- Hyperparameters are selected using cross-validation on training data
- The final test set remains untouched until final evaluation

## 12. Reproducibility

The preprocessing pipeline is implemented in:

```text
src/prepare_description_dataset.py