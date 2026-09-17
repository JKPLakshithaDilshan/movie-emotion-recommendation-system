from pathlib import Path
import nbformat as nbf

OUT = Path("IT25100112/IT25100112_logistic_regression.ipynb")
nb = nbf.v4.new_notebook()
cells = []

def md(text): cells.append(nbf.v4.new_markdown_cell(text.strip()))
def code(text): cells.append(nbf.v4.new_code_cell(text.strip()))

md(r'''# IT25100112 – Logistic Regression

**Student:** Karunarathne H. L. N. A.  
**IT number:** IT25100112  
**Project:** Emotion-Aware Movie Recommendation System  
**Primary preprocessing responsibility:** Handling missing data  
**Assigned model:** Logistic Regression  

This notebook independently demonstrates missing-data handling, categorical encoding, outlier treatment, scaling, feature engineering, feature selection, dimensionality reduction, EDA, leakage-safe model tuning, cross-validation, and final evaluation.''')

md(r'''## 1. Reproducibility and imports

Random state 42 is used throughout. All learned preprocessing steps used for modelling are fitted only on training folds through a scikit-learn pipeline.''')

code(r'''from pathlib import Path
import json
import warnings

import cloudpickle
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid", context="notebook")

PROJECT_ROOT = Path.cwd()

if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if not (PROJECT_ROOT / "data").exists():
    raise FileNotFoundError(
        "Could not locate the repository root containing the data directory."
    )

DATA_PATH = (
    PROJECT_ROOT
    / "data/processed/modelling_dataset_7class.csv"
)

MEMBER_DIR = PROJECT_ROOT / "IT25100112"
EDA_DIR = MEMBER_DIR / "results/eda_visualizations"
MODEL_FIG_DIR = MEMBER_DIR / "results/model_visualizations"
OUTPUT_DIR = MEMBER_DIR / "results/outputs"
LOG_DIR = MEMBER_DIR / "results/logs"
for directory in [EDA_DIR, MODEL_FIG_DIR, OUTPUT_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print("Random state:", RANDOM_STATE)
print("Dataset path:", DATA_PATH)''')

md(r'''## 2. Load and understand the assigned dataset

The modelling unit is one unique movie description. The target is one of seven emotions. The `surprise` class remains in the full EDA dataset but is excluded here because it has only three independent descriptions, which is insufficient for reliable stratified validation.''')

code(r'''df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Duplicate descriptions:", int(df["Description"].duplicated().sum()))
display(df.head(3))
display(df.dtypes.to_frame("dtype"))''')

md(r'''## 3. Missing-data handling — primary specialization

Missing values can cause vectorizers, encoders, scalers, and models to fail or silently lose records. The audit covers literal nulls, empty strings, and unavailable genre lists. Empty genre information is represented by the explicit category `unknown_genre`, preserving the observation instead of deleting it.

For leakage prevention, the modelling pipeline also contains `SimpleImputer`: most-frequent imputation for categorical inputs and median imputation for numerical inputs. These statistics are learned from training folds only.''')

code(r'''blank_counts = {}
for column in df.select_dtypes(include=["object", "string"]).columns:
    blank_counts[column] = int(df[column].astype("string").str.strip().eq("").sum())

missing_audit = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "blank_count": pd.Series(blank_counts),
}).fillna(0).astype(int)

print("Total literal missing values:", int(df.isna().sum().sum()))
print("Total blank strings:", int(missing_audit["blank_count"].sum()))
print("Unknown-genre rows:", int(df["genres_text"].eq("unknown_genre").sum()))
display(missing_audit)

# Independent handling demonstration
demo_missing = df[["mean_rating", "genres_text"]].copy()
demo_missing.loc[demo_missing.index[0], "mean_rating"] = np.nan
demo_missing.loc[demo_missing.index[1], "genres_text"] = np.nan

numeric_imputer_demo = SimpleImputer(strategy="median")
category_imputer_demo = SimpleImputer(strategy="most_frequent")
demo_missing[["mean_rating"]] = numeric_imputer_demo.fit_transform(demo_missing[["mean_rating"]])
demo_missing["genres_text"] = (
    category_imputer_demo
    .fit_transform(demo_missing[["genres_text"]])
    .ravel()
)
print("Demonstration missing values after imputation:", int(demo_missing.isna().sum().sum()))''')

md(r'''## 4. Exploratory data analysis

All figures are saved in the student's own results folder. Each visualization is followed by an interpretation and its influence on modelling.''')

md('''### EDA 1 — Emotion class distribution''')
code(r'''class_counts = df["emotion"].value_counts().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
class_counts.plot(kind="barh", ax=ax, color=sns.color_palette("viridis", len(class_counts)))
ax.set_title("Emotion Class Distribution")
ax.set_xlabel("Number of descriptions")
ax.set_ylabel("Emotion")
for container in ax.containers:
    ax.bar_label(container, padding=3)
fig.tight_layout()
fig.savefig(EDA_DIR / "01_emotion_class_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
display(class_counts.to_frame("count"))''')
md(r'''**Interpretation:** Sadness is the majority class (817 records), while disgust is the smallest modelled class (73). Accuracy alone could hide weak minority-class performance. Therefore, stratification, class-weight comparison, macro F1, and per-class recall are required.''')

md('''### EDA 2 — Mean-rating histogram''')
code(r'''fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df["mean_rating"], bins=20, kde=True, color="#2563eb", ax=ax)
ax.set_title("Distribution of Mean Movie Ratings")
ax.set_xlabel("Mean rating")
ax.set_ylabel("Frequency")
fig.tight_layout()
fig.savefig(EDA_DIR / "02_mean_rating_histogram.png", dpi=300, bbox_inches="tight")
plt.show()
print(df["mean_rating"].describe())''')
md(r'''**Interpretation:** Ratings occupy a different numerical scale from count and length features. Standardization is therefore applied inside the pipeline so coefficients are not dominated merely by feature magnitude.''')

md('''### EDA 3 — Mean-rating boxplot by emotion''')
code(r'''order = df.groupby("emotion")["mean_rating"].median().sort_values().index
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df, x="emotion", y="mean_rating", order=order, palette="Set2", ax=ax)
ax.set_title("Mean Rating by Emotion")
ax.set_xlabel("Emotion")
ax.set_ylabel("Mean rating")
ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(EDA_DIR / "03_mean_rating_boxplot_by_emotion.png", dpi=300, bbox_inches="tight")
plt.show()''')
md(r'''**Interpretation:** Rating distributions overlap substantially across emotions, so rating alone cannot separate the classes. It is retained as supporting information, while TF-IDF description features remain the primary signal. Visible extreme values motivate robust IQR clipping.''')

md('''### EDA 4 — Emotion class percentages''')
code(r'''class_percent = df["emotion"].value_counts(normalize=True).mul(100).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=class_percent.values, y=class_percent.index, palette="mako", ax=ax)
ax.set_title("Percentage Share of Each Emotion Class")
ax.set_xlabel("Percentage (%)")
ax.set_ylabel("Emotion")
for i, value in enumerate(class_percent.values):
    ax.text(value + 0.3, i, f"{value:.1f}%", va="center")
fig.tight_layout()
fig.savefig(EDA_DIR / "04_emotion_class_percentage.png", dpi=300, bbox_inches="tight")
plt.show()''')
md(r'''**Interpretation:** Nearly two-fifths of observations are sadness, confirming severe imbalance. Macro F1 is selected as the tuning metric because it gives every emotion equal importance.''')

md(r'''## 5. Feature engineering

The notebook uses description text, genre category, rating summaries, description lengths, genre count, rating range, and log-transformed review-popularity measures. These features represent semantic content, categorical context, quality, complexity, and popularity.''')

code(r'''numeric_features = [
    "mean_rating",
    "median_rating",
    "genre_count",
    "description_word_count",
    "description_character_count",
    "rating_range",
    "log_review_count",
    "log_unique_review_count",
]
categorical_features = ["genres_text"]
text_feature = "Description"
target = "emotion"

model_columns = [text_feature] + categorical_features + numeric_features
X = df[model_columns].copy()
y = df[target].copy()

feature_summary = df[numeric_features].describe().T
display(feature_summary)
print("Input columns:", model_columns)''')

md(r'''## 6. Outlier detection and treatment

The IQR rule flags observations below $Q_1-1.5(IQR)$ or above $Q_3+1.5(IQR)$. Deleting valid movies could further damage minority classes, so numeric outliers are winsorized (clipped) rather than removed. The custom transformer learns limits only from training data.''')

code(r'''outlier_rows = []
for column in numeric_features:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((df[column] < lower) | (df[column] > upper)).sum())
    outlier_rows.append({"feature": column, "lower_bound": lower, "upper_bound": upper, "outlier_count": count})

outlier_table = pd.DataFrame(outlier_rows).sort_values("outlier_count", ascending=False)
display(outlier_table)

class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.q1_ = np.nanquantile(values, 0.25, axis=0)
        self.q3_ = np.nanquantile(values, 0.75, axis=0)
        iqr = self.q3_ - self.q1_
        self.lower_ = self.q1_ - self.factor * iqr
        self.upper_ = self.q3_ + self.factor * iqr
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_, self.upper_)''')

md(r'''## 7. Categorical encoding and target encoding

`genres_text` is encoded with `OneHotEncoder(handle_unknown='ignore')`, allowing unseen test categories without failure. `LabelEncoder` is demonstrated for the target, although scikit-learn Logistic Regression can train directly on string labels.''')

code(r'''label_encoder = LabelEncoder()
encoded_target = label_encoder.fit_transform(y)
target_mapping = pd.DataFrame({
    "emotion": label_encoder.classes_,
    "encoded_value": range(len(label_encoder.classes_)),
})
display(target_mapping)

genre_encoder_demo = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
genre_matrix_demo = genre_encoder_demo.fit_transform(df[["genres_text"]])
print("Genre encoding shape:", genre_matrix_demo.shape)
print("Number of encoded genre categories:", len(genre_encoder_demo.categories_[0]))''')

md(r'''## 8. Normalization and scaling

StandardScaler is demonstrated below to show standardization. In the final pipeline, MinMaxScaler is used after median imputation and IQR clipping so numeric values remain non-negative for the chi-squared selector. All scaling parameters are fitted only on training folds.''')

code(r'''scaler_demo = StandardScaler()
scaled_demo = scaler_demo.fit_transform(df[numeric_features])
scaled_summary = pd.DataFrame(scaled_demo, columns=numeric_features).agg(["mean", "std"]).T
display(scaled_summary.round(4))''')

md(r'''## 9. Stratified train/test split

The 20% final test set remains untouched during tuning. Stratification preserves the seven-class proportions.''')

code(r'''X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

split_summary = pd.concat([
    y_train.value_counts().rename("train_count"),
    y_test.value_counts().rename("test_count"),
], axis=1).fillna(0).astype(int)
display(split_summary)
print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)''')

md(r'''## 10. Leakage-safe preprocessing pipeline

- Description: TF-IDF unigrams or unigrams+bigrams.
- Genre: most-frequent imputation and one-hot encoding.
- Numeric: median imputation, IQR clipping, and min-max scaling.
- Combined features: chi-squared selection.
- Classifier: multinomial Logistic Regression.

Every learned step is nested in `GridSearchCV`, so each validation fold learns transformations only from its training portion.''')

code(r'''text_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing_description")),
    ("flatten", __import__("sklearn").preprocessing.FunctionTransformer(lambda x: np.asarray(x).ravel(), validate=False)),
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", min_df=2, max_df=0.95, sublinear_tf=True)),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
])

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("outliers", IQRClipper(factor=1.5)),
    # Min-max scaling keeps values non-negative, which is required
    # by the downstream chi-squared feature selector.
    ("scaler", MinMaxScaler()),
])

preprocessor = ColumnTransformer([
    ("text", text_pipeline, [text_feature]),
    ("genre", categorical_pipeline, categorical_features),
    ("numeric", numeric_pipeline, numeric_features),
])

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("select", SelectKBest(score_func=chi2, k=1000)),
    ("model", LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )),
])

print(pipeline)''')

md(r'''## 11. Feature selection

Chi-squared selection keeps features most associated with the emotion target, reducing noise and training cost. It is fitted inside cross-validation. The search compares 500, 1000, and 1500 selected features.''')

md(r'''## 12. Dimensionality reduction demonstration

TruncatedSVD is appropriate for sparse TF-IDF data. It is demonstrated on the training descriptions only. It is not forced into the final Logistic Regression pipeline because supervised chi-squared selection preserves interpretable terms and may perform better for linear text classification.''')

code(r'''svd_tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    min_df=2,
    max_df=0.95,
    ngram_range=(1, 2),
    max_features=5000,
    sublinear_tf=True,
)
train_tfidf = svd_tfidf.fit_transform(X_train[text_feature])
svd = TruncatedSVD(n_components=100, random_state=RANDOM_STATE)
train_svd = svd.fit_transform(train_tfidf)
cum_variance = np.cumsum(svd.explained_variance_ratio_)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, len(cum_variance) + 1), cum_variance, color="#7c3aed")
ax.set_title("TruncatedSVD Cumulative Explained Variance")
ax.set_xlabel("Number of components")
ax.set_ylabel("Cumulative explained variance")
fig.tight_layout()
fig.savefig(MODEL_FIG_DIR / "svd_explained_variance.png", dpi=300, bbox_inches="tight")
plt.show()

print("Original training TF-IDF shape:", train_tfidf.shape)
print("Reduced training shape:", train_svd.shape)
print("Variance explained by 100 components:", round(float(cum_variance[-1]), 4))''')

md(r'''## 13. Hyperparameter tuning and model varieties

The search compares regularization strengths, class weighting, unigram versus unigram-bigram TF-IDF, and feature-selection sizes. Macro F1 is optimized due to class imbalance.''')

code(r'''cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

param_grid = {
    "preprocess__text__tfidf__ngram_range": [(1, 1), (1, 2)],
    "select__k": [500, 1000, 1500],
    "model__C": [0.1, 1.0, 10.0],
    "model__class_weight": [None, "balanced"],
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
    refit=True,
)

grid_search.fit(X_train, y_train)
print("Candidates trained:", len(grid_search.cv_results_["params"]))
print("Best CV macro F1:", round(grid_search.best_score_, 4))
print("Best parameters:")
print(json.dumps(grid_search.best_params_, indent=2, default=str))''')

md(r'''## 14. Compare model varieties''')

code(r'''cv_results = pd.DataFrame(grid_search.cv_results_)
comparison_columns = [
    "param_model__C",
    "param_model__class_weight",
    "param_preprocess__text__tfidf__ngram_range",
    "param_select__k",
    "mean_train_score",
    "mean_test_score",
    "std_test_score",
    "rank_test_score",
]
model_comparison = (
    cv_results[comparison_columns]
    .sort_values("rank_test_score")
    .reset_index(drop=True)
)
display(model_comparison.head(15))
model_comparison.to_csv(OUTPUT_DIR / "logistic_regression_model_comparison.csv", index=False)''')

md(r'''## 15. Final untouched-test evaluation''')

code(r'''best_model = grid_search.best_estimator_
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

metrics = {
    "train_accuracy": accuracy_score(y_train, y_train_pred),
    "test_accuracy": accuracy_score(y_test, y_test_pred),
    "macro_precision": precision_score(y_test, y_test_pred, average="macro", zero_division=0),
    "macro_recall": recall_score(y_test, y_test_pred, average="macro", zero_division=0),
    "macro_f1": f1_score(y_test, y_test_pred, average="macro", zero_division=0),
    "weighted_f1": f1_score(y_test, y_test_pred, average="weighted", zero_division=0),
    "best_cv_macro_f1": grid_search.best_score_,
}

metrics_df = pd.DataFrame([metrics]).T.rename(columns={0: "score"})
display(metrics_df.round(4))
metrics_df.to_csv(OUTPUT_DIR / "logistic_regression_metrics.csv")

report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).T
display(report_df.round(4))
report_df.to_csv(OUTPUT_DIR / "logistic_regression_classification_report.csv")''')

md(r'''## 16. Confusion matrix''')

code(r'''labels = sorted(y.unique())
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_test_pred,
    labels=labels,
    display_labels=labels,
    cmap="Blues",
    xticks_rotation=35,
    ax=ax,
    colorbar=False,
)
ax.set_title("Logistic Regression Confusion Matrix")
fig.tight_layout()
fig.savefig(MODEL_FIG_DIR / "logistic_regression_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()''')

md(r'''## 17. Five-fold cross-validation reliability

The tuned pipeline is evaluated using multiple metrics over the training data. Mean and standard deviation show expected generalization and stability.''')

code(r'''scoring = {
    "accuracy": "accuracy",
    "macro_precision": "precision_macro",
    "macro_recall": "recall_macro",
    "macro_f1": "f1_macro",
    "weighted_f1": "f1_weighted",
}
cv_scores = cross_validate(best_model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
cv_summary = pd.DataFrame({
    metric.replace("test_", ""): {
        "mean": np.mean(values),
        "std": np.std(values),
    }
    for metric, values in cv_scores.items()
    if metric.startswith("test_")
}).T
display(cv_summary.round(4))
cv_summary.to_csv(OUTPUT_DIR / "logistic_regression_cross_validation.csv")''')

md(r'''## 18. Save predictions and trained model''')

code(r'''predictions = pd.DataFrame({
    "actual_emotion": y_test.reset_index(drop=True),
    "predicted_emotion": pd.Series(y_test_pred),
})
predictions.to_csv(OUTPUT_DIR / "logistic_regression_test_predictions.csv", index=False)
model_path = (
    OUTPUT_DIR
    / "IT25100112_best_logistic_regression.pkl"
)

with open(model_path, "wb") as model_file:
    cloudpickle.dump(best_model, model_file)

summary = {
    "student_id": "IT25100112",
    "student_name": "Karunarathne H. L. N. A.",
    "model": "Logistic Regression",
    "random_state": RANDOM_STATE,
    "best_parameters": grid_search.best_params_,
    "metrics": {key: float(value) for key, value in metrics.items()},
}
with open(OUTPUT_DIR / "logistic_regression_summary.json", "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2, default=str)

print("Saved model and outputs to:", OUTPUT_DIR)''')

md(r'''## 19. Conclusion, limitations, and improvements

**Model selection:** The final Logistic Regression variety is selected using five-fold cross-validated macro F1, not test-set performance. This protects the test set and gives equal importance to minority classes.

**Interpretation:** Compare macro F1 with weighted F1 and inspect the confusion matrix. A much lower macro F1 indicates weaker minority-class prediction. The `disgust` row deserves particular attention because it has the fewest training examples.

**Potential limitations:**

- Class imbalance, especially the small disgust class.
- Only 2,057 independent descriptions.
- Emotion labels may be subjective.
- Descriptions may contain incomplete summaries.
- Genre combinations encoded as categories can be sparse.
- The excluded surprise class cannot be predicted by the supervised model.

**Possible improvements:**

- Collect more independently labelled descriptions for minority emotions.
- Use multilabel genre encoding.
- Compare calibrated linear SVM and transformer embeddings.
- Conduct fairness/error analysis across genres.
- Add explainability using highest positive class coefficients.

## 20. Individual reflection

This implementation demonstrates that reliable evaluation depends on data quality, leakage-safe preprocessing, stratification, and metrics suited to imbalance. Logistic Regression is a strong interpretable baseline for high-dimensional TF-IDF text, but class-level analysis is essential before deployment.

## AI tool usage declaration

Generative AI assistance was used for code structuring, debugging suggestions, documentation organization, and explanation refinement. All code must be executed, checked, modified where necessary, and understood by the student before submission. The student remains responsible for every result and interpretation.''')

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python (Movie Emotion ML)", "language": "python", "name": "movie-emotion-ml"},
    "language_info": {"name": "python", "version": "3.14"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"Created {OUT} with {len(cells)} cells")
