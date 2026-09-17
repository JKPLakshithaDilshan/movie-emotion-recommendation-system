# Data Dictionary

## 1. Raw dataset

**File:** `data/raw/Movies_Reviews_modified_version1.csv`  
**Shape:** 46,173 rows and 8 columns

| Column | Type | Description | Project decision |
|---|---|---|---|
| `Unnamed: 0` | Integer | Sequential exported row number from 0 to 46,172 | Removed because it has no predictive meaning |
| `Ratings` | Float | Movie or review rating on a 1–10 scale | Validated, aggregated, and used as a supporting numerical feature |
| `Reviews` | Text | English audience-review text | Audited but excluded as the primary classifier input because repeated reviews have conflicting emotion labels |
| `movie_name` | Text | Movie-title string | Retained for display and recommendation output |
| `Resenhas` | Text | Portuguese translation of the English review | Excluded from modelling because it duplicates the review information in another language |
| `genres` | Categorical text | Genre list stored as a string | Parsed and encoded as genre features |
| `Description` | Text | Movie plot description or summary | Primary NLP input and independent observation key |
| `emotion` | Categorical target | Assigned emotion label | Target variable |

## 2. Full processed dataset

**File:** `data/processed/description_level_dataset.csv`  
**Shape:** 2,060 rows and 19 columns  
**Observation:** One unique movie description

| Column | Type | Description | Modelling role |
|---|---|---|---|
| `Description` | Text | Unique movie description or plot summary | Primary TF-IDF text feature |
| `emotion` | Categorical | Emotion assigned to the unique description | Classification target |
| `movie_name` | Text | First representative movie name associated with the description | Recommendation display field |
| `all_movie_names` | Text | All unique movie names associated with the description | Audit and display field |
| `movie_name_count` | Integer | Number of unique movie names associated with the description | Data-quality indicator |
| `genres` | Categorical text | Original stored genre-list string | Source categorical field |
| `mean_rating` | Float | Mean rating across rows sharing the description | Supporting numerical feature |
| `median_rating` | Float | Median rating across rows sharing the description | Supporting numerical feature |
| `minimum_rating` | Float | Minimum rating associated with the description | Supporting numerical feature |
| `maximum_rating` | Float | Maximum rating associated with the description | Supporting numerical feature |
| `review_count` | Integer | Number of rows or reviews associated with the description | Popularity feature and outlier-analysis field |
| `unique_review_count` | Integer | Number of unique English review texts associated with the description | Supporting popularity feature |
| `genre_list` | Text representation of list | Parsed genre values saved in CSV-compatible form | Interpretation and reproducibility field |
| `genre_count` | Integer | Number of genres assigned to the description | Engineered numerical feature |
| `genres_text` | Text | Normalized genre tokens separated by spaces | Encoded categorical-text feature |
| `description_word_count` | Integer | Number of words in the description | Engineered length feature |
| `description_character_count` | Integer | Number of characters in the description | Engineered length feature |
| `log_review_count` | Float | `log(1 + review_count)` | Skew-reduced popularity feature |
| `log_unique_review_count` | Float | `log(1 + unique_review_count)` | Skew-reduced popularity feature |

## 3. Seven-class modelling dataset

**File:** `data/processed/modelling_dataset_7class.csv`  
**Shape:** 2,057 rows and 19 columns

This dataset contains the same fields as the full processed dataset but excludes the three independent `surprise` records.

### Included target classes

| Class | Independent records |
|---|---:|
| Sadness | 817 |
| Joy | 400 |
| Anticipation | 249 |
| Optimism | 237 |
| Fear | 150 |
| Anger | 131 |
| Disgust | 73 |

### Excluded target class

| Class | Independent records | Decision |
|---|---:|---|
| Surprise | 3 | Retained in EDA but excluded from supervised training because reliable stratified validation is impossible |

## 4. Missing-information policy

The raw dataset contained no literal null values, but 35 unique descriptions had an empty genre list represented by:

```text
[]