# Emotion-Aware Movie Recommendation System

An end-to-end machine learning project that predicts emotional categories from movie descriptions and supports movie recommendations based on emotion, genre, and rating.

## Module

**Module:** IT2011 – Artificial Intelligence and Machine Learning  
**Academic year:** Year 2, Semester 1 – 2026  
**Assignment:** Design, Implement, and Evaluate AI/ML Solutions for a Real-World Problem Using a Given Dataset

## Problem Statement

Users often struggle to identify movies that match their current mood. Conventional recommendation systems commonly focus on ratings, popularity, and viewing history but may not represent the emotional characteristics of movie content.

This project develops an NLP-based multi-class classifier that predicts a movie’s emotion from its description. The predicted emotion is then combined with genre and rating information to support emotion-aware movie recommendations.

## Machine Learning Task

**Task type:** Supervised multi-class classification

**Primary input:**

- Movie description

**Supporting inputs:**

- Genre information
- Aggregated rating features
- Description-length features
- Review-popularity features

**Target variable:**

- Emotion

**Modelled classes:**

- Anger
- Anticipation
- Disgust
- Fear
- Joy
- Optimism
- Sadness

The `surprise` class is retained for EDA but excluded from supervised training because it contains only three independent description-level observations.

## Assigned Dataset

**Dataset:** IMDb Movie Reviews – Genres, Descriptions and Emotions  
**Source:** [Kaggle dataset](https://www.kaggle.com/datasets/fahadrehman07/movie-reviews-and-emotion-dataset)  
**Raw format:** CSV  
**Raw shape:** 46,173 rows and 8 columns  
**Raw file size:** Approximately 129 MB  
**SHA-256:** `81ca8d80d2285b75e5f14cb0e7daefe532118ac0155583df8c6b9d1296bdc580`

The raw dataset is tracked using Git LFS.

## Data-Quality Findings

The initial audit found:

- No literal null values
- No whitespace-only values
- 2,192 duplicates after removing the artificial index
- 19,316 unique English reviews
- 3,188 repeated reviews associated with conflicting emotion labels
- 2,060 unique and label-consistent movie descriptions
- 35 descriptions with empty genre lists
- Severe class imbalance
- Only three independent `surprise` records

Because repeated reviews contained conflicting targets, the final modelling unit is one unique movie description.

## Processed Datasets

### Full EDA Dataset

```text
data/processed/description_level_dataset.csv
```

- 2,060 independent descriptions
- Eight emotion classes
- 19 columns
- Zero missing values
- Zero duplicated descriptions
- Used for complete EDA and data-quality reporting

### Seven-Class Modelling Dataset

```text
data/processed/modelling_dataset_7class.csv
```

- 2,057 independent descriptions
- Seven emotion classes
- 19 columns
- Zero missing values
- Zero duplicated descriptions
- Used for supervised model training and evaluation

## Group Members and Assigned Models

| Student ID | Student name | Assigned model |
|---|---|---|
| IT25100112 | Karunarathne H. L. N. A. | Logistic Regression |
| IT25100083 | Unawatuna H. M. I. D. | Linear Support Vector Machine |
| IT25100078 | Thujikoshan Y. | Multinomial Naive Bayes |
| IT25100090 | Hettiarachchi H. C. N. G. | Decision Tree |
| IT25100111 | Chandrasekara C. M. T. D. | Random Forest |
| IT25100149 | Rajapaksha P. A. D. S. | Multi-Layer Perceptron |

## Individual Requirements

Every student independently handles and presents:

- Missing-data handling
- Categorical-variable encoding
- Outlier detection and removal or treatment
- Normalization or scaling
- Feature engineering
- Feature selection
- Dimensionality reduction
- At least three interpreted EDA visualizations
- Individual model implementation
- Hyperparameter tuning
- Model-variety comparison
- Evaluation and interpretation
- Limitations and improvements

The shared preprocessing script does not replace the individual requirement. Each student must show executable code, output, justification, and interpretation in their own notebook.

See [Individual member responsibilities](documentation/member_responsibilities.md).

## Repository Structure

```text
movie-emotion-recommendation-system/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── documentation/
│   ├── proposal/
│   ├── references/
│   ├── specifications/
│   ├── data_dictionary.md
│   ├── dataset_audit.md
│   └── member_responsibilities.md
├── src/
│   └── prepare_description_dataset.py
├── IT25100112/
├── IT25100083/
├── IT25100078/
├── IT25100090/
├── IT25100111/
├── IT25100149/
├── final_group_results/
├── final_group_report/
├── models/
├── presentations/
├── requirements.txt
└── README.md
```

## Individual Notebook Names

```text
IT25100112/IT25100112_logistic_regression.ipynb
IT25100083/IT25100083_linear_svm.ipynb
IT25100078/IT25100078_naive_bayes.ipynb
IT25100090/IT25100090_decision_tree.ipynb
IT25100111/IT25100111_random_forest.ipynb
IT25100149/IT25100149_mlp_classifier.ipynb
```

Each notebook includes the student’s:

- Name and IT number
- Assigned model
- Individual preprocessing
- EDA visualizations
- Feature engineering
- Feature selection
- Dimensionality reduction
- Hyperparameter tuning
- Model evaluation
- Interpretation
- Limitations
- Reflection

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/JKPLakshithaDilshan/movie-emotion-recommendation-system.git

cd movie-emotion-recommendation-system
```

### 2. Install Git LFS

On macOS with Homebrew:

```bash
brew install git-lfs

git lfs install

git lfs pull
```

### 3. Create a Virtual Environment

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip

python -m pip install -r requirements.txt
```

### 5. Register the Jupyter Kernel

```bash
python -m ipykernel install \
  --user \
  --name movie-emotion-ml \
  --display-name "Python (Movie Emotion ML)"
```

## Regenerate the Processed Datasets

Run:

```bash
python src/prepare_description_dataset.py
```

Expected results:

```text
Full description-level dataset: 2,060 rows
Seven-class modelling dataset: 2,057 rows
Unknown-genre records: 35
Remaining missing values: 0
```

## Start JupyterLab

```bash
jupyter lab
```

Select the following kernel:

```text
Python (Movie Emotion ML)
```

## Preprocessing Strategy

The project applies:

- Missing-value and blank-string auditing
- Empty-genre encoding using `unknown_genre`
- Artificial index removal
- Exact duplicate removal
- Description-level aggregation
- Rating-range validation
- IQR-based outlier analysis
- Log transformation of skewed count features
- Categorical encoding
- Numerical scaling
- TF-IDF text vectorization
- Chi-squared feature selection
- TruncatedSVD dimensionality reduction

All learned transformations are fitted using training data only to prevent data leakage.

## Exploratory Data Analysis

Every student produces at least three interpreted visualizations. The group analysis examines:

- Emotion class distribution
- Rating distribution
- Description-length distribution
- Review-count distribution
- Genre frequencies
- Numeric-feature correlations
- Outliers
- Relationships between features and emotion classes
- Class imbalance

Each visualization must include:

- A meaningful title
- Labelled axes
- A readable legend where applicable
- Saved PNG output
- Written interpretation
- Explanation of its effect on preprocessing or modelling decisions

## Feature Engineering

Engineered features include:

- Description word count
- Description character count
- Genre count
- Mean rating
- Median rating
- Minimum rating
- Maximum rating
- Review count
- Unique review count
- Log-transformed review count
- Log-transformed unique-review count
- TF-IDF description features
- Encoded genre features

## Feature Selection

Feature selection includes:

- TF-IDF `min_df`
- TF-IDF `max_df`
- TF-IDF `max_features`
- Chi-squared `SelectKBest`
- Removal of non-informative features

All feature-selection methods are fitted only using training data.

## Dimensionality Reduction

The project uses `TruncatedSVD` because TF-IDF creates a sparse, high-dimensional feature matrix.

The number of SVD components is justified using:

- Explained variance
- Cross-validation performance
- Computational cost
- Suitability for the assigned model

## Model Training

The six individually assigned algorithms are:

1. Logistic Regression
2. Linear Support Vector Machine
3. Multinomial Naive Bayes
4. Decision Tree
5. Random Forest
6. Multi-Layer Perceptron

Each student trains and compares multiple varieties of their assigned model by changing relevant hyperparameters.

## Evaluation Strategy

The project uses:

- Stratified train-test splitting
- Five-fold stratified cross-validation
- Grid search or another justified tuning approach
- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Weighted F1-score
- Per-class evaluation
- Confusion matrices
- Training-versus-test comparison
- Overfitting and underfitting analysis

Macro F1 is emphasized because the target classes are imbalanced.

## Recommendation Component

After identifying the best-performing emotion classifier, the recommendation component will:

1. Accept a user mood or emotional preference
2. Identify the matching emotion category
3. Filter movies by predicted emotion
4. Apply optional genre preferences
5. Apply an optional minimum rating
6. Remove repeated movie titles
7. Rank and return suitable movies

The system is an emotion-aware content-based recommendation approach. It is not collaborative filtering because the dataset does not contain user IDs or user–movie interaction histories.

## Final Deliverables

- Six individual notebooks
- At least three interpreted EDA visualizations per student
- Six individually assigned model implementations
- Multiple tuned model varieties per student
- Individual evaluation and conclusions
- Six individual PDF reports named with Student IDs
- One final group report
- Final six-model comparison table
- Saved best-performing model
- Emotion-aware recommendation demonstration
- Viva preparation materials
- CourseWeb group-deliverable ZIP

## Report Structure

The individual and group reports will address:

1. Introduction and Problem Statement
2. Dataset Description
3. Data Preprocessing and EDA
4. Model Design and Implementation
5. Model Evaluation and Comparison
6. Ethical Considerations and Bias Mitigation
7. AI Tool Usage Declaration and Transparency
8. Reflections and Lessons Learned
9. References

## Documentation

- [Dataset audit](documentation/dataset_audit.md)
- [Data dictionary](documentation/data_dictionary.md)
- [Individual member responsibilities](documentation/member_responsibilities.md)

## Reproducibility

Standard random state:

```text
42
```

All submitted results must come from executed code. Model scores, charts, comparisons, and conclusions must not be manually invented.

## Ethical Considerations

The project considers:

- Target-class imbalance
- Underrepresentation of `surprise`
- Cultural and language bias
- Translation-related meaning changes
- IMDb reviewer-selection bias
- Incorrect emotion predictions
- Recommendation filter bubbles
- Explainability limitations
- Potential misuse of mood-related predictions

Predicted emotions are content labels and must not be represented as psychological assessments of users.

## AI Tool Transparency

Any use of ChatGPT, Claude, GitHub Copilot, Gemini, or another AI tool must be declared in the final reports.

For each AI tool, students must state:

- Tool name and version, if known
- Tasks supported by the tool
- Extent of use
- How generated suggestions were tested
- How outputs were verified or modified

Students remain responsible for:

- Understanding all submitted code
- Testing AI-assisted suggestions
- Verifying outputs
- Correcting errors
- Explaining all decisions during the viva