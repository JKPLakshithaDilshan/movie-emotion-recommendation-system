from pathlib import Path
import nbformat as nbf

OUT = Path("IT25100111/IT25100111_random_forest.ipynb")
nb = nbf.v4.new_notebook()
cells = []

def md(text): cells.append(nbf.v4.new_markdown_cell(text.strip()))
def code(text): cells.append(nbf.v4.new_code_cell(text.strip()))

md(r'''# IT25100111 – Random Forest

**Student:** Chandrasekara C. M. T. D.  
**IT number:** IT25100111  
**Project:** Emotion-Aware Movie Recommendation System  
**Primary preprocessing responsibility:** Dimensionality reduction  
**Assigned model:** Random Forest  

This notebook independently demonstrates missing-data handling, categorical encoding, outlier treatment, scaling, feature engineering, feature selection, dimensionality reduction, EDA, leakage-safe model tuning, cross-validation, and final evaluation.''')

md(r'''## 1. Reproducibility and imports

Random state 42 is used throughout. All learned preprocessing steps used for modelling are fitted only on training folds through a scikit-learn pipeline.''')

code(r'''from pathlib import Path
import json
import warnings

import cloudpickle
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
from sklearn.ensemble import RandomForestClassifier
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
    raise FileNotFoundError("Could not locate the repository root containing the data directory.")
DATA_PATH = PROJECT_ROOT / "data/processed/modelling_dataset_7class.csv"
MEMBER_DIR = PROJECT_ROOT / "IT25100111"
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

md(r'''## 3. Missing-data handling

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
demo_missing["genres_text"] = category_imputer_demo.fit_transform(demo_missing[["genres_text"]]).ravel()
print("Demonstration missing values after imputation:", int(demo_missing.isna().sum().sum()))''')

md(r'''## 4. Exploratory data analysis

All figures are saved in the student's own results folder. Each visualization is followed by an interpretation and its influence on modelling.''')

md('''### EDA 1 — Raw review-count distribution''')
code(r'''fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df["review_count"], bins=35, kde=True, color="#2563eb", ax=ax)
ax.set_title("Raw Review-Count Distribution")
ax.set_xlabel("Reviews per description")
ax.set_ylabel("Frequency")
fig.tight_layout()
fig.savefig(EDA_DIR / "01_raw_review_count_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
display(df["review_count"].describe().to_frame("value"))''')
md(r'''**Interpretation:** Raw review counts are strongly right-skewed. A small number of popular descriptions have many reviews, so the raw count could dominate smaller numerical features without transformation and scaling.''')

md('''### EDA 2 — Log-transformed review-count distribution''')
code(r'''fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df["log_review_count"], bins=30, kde=True, color="#14b8a6", ax=ax)
ax.set_title("Log-Transformed Review-Count Distribution")
ax.set_xlabel("log(1 + review count)")
ax.set_ylabel("Frequency")
fig.tight_layout()
fig.savefig(EDA_DIR / "02_log_review_count_distribution.png", dpi=300, bbox_inches="tight")
plt.show()''')
md(r'''**Interpretation:** The `log1p` transformation compresses the long upper tail while preserving ordering. This produces a more stable popularity feature without deleting genuinely popular movies.''')

md('''### EDA 3 — Review count by emotion''')
code(r'''order = df.groupby("emotion")["review_count"].median().sort_values().index
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df, x="emotion", y="review_count", order=order, palette="Set2", ax=ax)
ax.set_title("Review Count by Emotion")
ax.set_xlabel("Emotion")
ax.set_ylabel("Reviews per description")
ax.tick_params(axis="x", rotation=30)
ax.set_yscale("log")
fig.tight_layout()
fig.savefig(EDA_DIR / "03_review_count_by_emotion.png", dpi=300, bbox_inches="tight")
plt.show()''')
md(r'''**Interpretation:** Popularity varies within every emotion and contains many high-count outliers. The strong overlap shows that review count cannot determine emotion alone, but it may add supporting information after log transformation.''')

md('''### EDA 4 — Mean rating versus review count''')
code(r'''fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x="log_review_count", y="mean_rating", hue="emotion", alpha=0.55, ax=ax)
ax.set_title("Mean Rating versus Log Review Count")
ax.set_xlabel("log(1 + review count)")
ax.set_ylabel("Mean rating")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(EDA_DIR / "04_mean_rating_vs_review_count.png", dpi=300, bbox_inches="tight")
plt.show()
print("Correlation:", round(df["log_review_count"].corr(df["mean_rating"]), 4))''')
md(r'''**Interpretation:** Mean rating and log popularity show substantial overlap across emotions and only limited linear association. Both remain supporting signals while description semantics provide the main predictive information.''')

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

`genres_text` is encoded with `OneHotEncoder(handle_unknown='ignore')`, allowing unseen test categories without failure. `LabelEncoder` is demonstrated for the target, although scikit-learn's Random Forest can train directly on string labels.''')

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

StandardScaler is demonstrated below to show standardization. In the final pipeline, MinMaxScaler follows median imputation and IQR clipping. Tree splits do not mathematically require scaling, but a consistent range prevents numeric magnitudes from dominating the combined representation before feature selection and SVD. All parameters are learned from training folds only.''')

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

- Description: TF-IDF unigrams and bigrams.
- Genre: most-frequent imputation and one-hot encoding.
- Numeric: median imputation, IQR clipping, and min-max scaling.
- Combined features: chi-squared selection followed by TruncatedSVD.
- Classifier: Random Forest trained on the manageable dense SVD representation.

Every learned step is nested in `GridSearchCV`, so each validation fold learns transformations only from its training portion.''')

code(r'''text_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing_description")),
    ("flatten", __import__("sklearn").preprocessing.FunctionTransformer(lambda x: np.asarray(x).ravel(), validate=False)),
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", min_df=2, max_df=0.95, ngram_range=(1, 2), sublinear_tf=True)),
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
    ("select", SelectKBest(score_func=chi2, k=1500)),
    ("svd", TruncatedSVD(n_components=100, random_state=RANDOM_STATE)),
    ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)),
])

print(pipeline)''')

md(r'''## 11. Feature selection

Chi-squared selection retains 1,500 non-negative features with the strongest target association before dimensionality reduction. It reduces noise and computational cost and is fitted inside every cross-validation fold.''')

md(r'''## 12. Dimensionality reduction — primary specialization

TruncatedSVD compresses high-dimensional sparse text and supporting features into 100 dense latent components. This makes tree-ensemble training manageable, reduces redundant vocabulary dimensions, and avoids converting the complete sparse matrix into an excessively large dense array. The standalone training-data demonstration below visualizes explained variance; the final pipeline independently fits SVD inside each training fold.''')

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

The search compares 100, 200, and 300 trees; depths of 20, 40, and unrestricted; square-root versus logarithmic feature sampling; and balanced versus unbalanced class weights. This produces 36 model varieties and 180 total fits under five-fold validation. Macro F1 is optimized because the classes are imbalanced.''')

code(r'''cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [20, 40, None],
    "model__max_features": ["sqrt", "log2"],
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
    "param_model__n_estimators",
    "param_model__max_depth",
    "param_model__max_features",
    "param_model__class_weight",
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
model_comparison.to_csv(OUTPUT_DIR / "random_forest_model_comparison.csv", index=False)''')

md(r'''## 15. Final untouched-test evaluation''')

code(r'''best_model = grid_search.best_estimator_
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

metrics = {
    "train_accuracy": accuracy_score(y_train, y_train_pred),
    "test_accuracy": accuracy_score(y_test, y_test_pred),
    "train_test_accuracy_gap": accuracy_score(y_train, y_train_pred) - accuracy_score(y_test, y_test_pred),
    "macro_precision": precision_score(y_test, y_test_pred, average="macro", zero_division=0),
    "macro_recall": recall_score(y_test, y_test_pred, average="macro", zero_division=0),
    "macro_f1": f1_score(y_test, y_test_pred, average="macro", zero_division=0),
    "weighted_f1": f1_score(y_test, y_test_pred, average="weighted", zero_division=0),
    "best_cv_macro_f1": grid_search.best_score_,
}

metrics_df = pd.DataFrame([metrics]).T.rename(columns={0: "score"})
display(metrics_df.round(4))
metrics_df.to_csv(OUTPUT_DIR / "random_forest_metrics.csv")

report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).T
display(report_df.round(4))
report_df.to_csv(OUTPUT_DIR / "random_forest_classification_report.csv")''')

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
ax.set_title("Random Forest Confusion Matrix")
fig.tight_layout()
fig.savefig(MODEL_FIG_DIR / "random_forest_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()''')

md(r'''### SVD-component importance

Because the forest is trained after TruncatedSVD, its importance values describe latent SVD components rather than individual original words. The chart is useful for detecting whether the ensemble relies on a small number of components, but it must not be interpreted as direct word importance.''')

code(r'''forest = best_model.named_steps["model"]
component_importance = pd.Series(
    forest.feature_importances_,
    index=[f"SVD component {i + 1}" for i in range(len(forest.feature_importances_))],
).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 7))
component_importance.head(20).sort_values().plot(kind="barh", color="#16a34a", ax=ax)
ax.set_title("Top 20 Random Forest SVD-Component Importances")
ax.set_xlabel("Impurity-based importance")
fig.tight_layout()
fig.savefig(MODEL_FIG_DIR / "random_forest_component_importance.png", dpi=300, bbox_inches="tight")
plt.show()
display(component_importance.head(20).to_frame("importance"))''')

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
cv_summary.to_csv(OUTPUT_DIR / "random_forest_cross_validation.csv")''')

md(r'''## 18. Save predictions and trained model''')

code(r'''predictions = pd.DataFrame({
    "actual_emotion": y_test.reset_index(drop=True),
    "predicted_emotion": pd.Series(y_test_pred),
})
predictions.to_csv(OUTPUT_DIR / "random_forest_test_predictions.csv", index=False)
model_path = OUTPUT_DIR / "IT25100111_best_random_forest.pkl"
with open(model_path, "wb") as model_file:
    cloudpickle.dump(best_model, model_file)

summary = {
    "student_id": "IT25100111",
    "student_name": "Chandrasekara C. M. T. D.",
    "model": "Random Forest",
    "random_state": RANDOM_STATE,
    "best_parameters": grid_search.best_params_,
    "metrics": {key: float(value) for key, value in metrics.items()},
}
with open(OUTPUT_DIR / "random_forest_summary.json", "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2, default=str)

print("Saved model and outputs to:", OUTPUT_DIR)''')

md(r'''## 19. Conclusion, limitations, and improvements

**Model selection:** The final Random Forest variety is selected using five-fold cross-validated macro F1, not test-set performance. This protects the test set and gives equal importance to minority classes.

**Interpretation:** Training and test accuracy are compared explicitly. A large positive gap still indicates overfitting even though aggregation is usually more stable than a single tree. Macro F1 and per-class recall reveal minority-class performance; the `disgust` row deserves particular attention because it has the fewest examples.

**Potential limitations:**

- Class imbalance, especially the small disgust class.
- Only 2,057 independent descriptions.
- Emotion labels may be subjective.
- Descriptions may contain incomplete summaries.
- Genre combinations encoded as categories can be sparse.
- Impurity-based component importance may favor frequently used continuous components.
- The excluded surprise class cannot be predicted by the supervised model.

**Possible improvements:**

- Collect more independently labelled descriptions for minority emotions.
- Compare multilabel genre encoding against the current combined-genre representation.
- Tune minimum leaf size and feature-selection/SVD dimensions.
- Compare Extra Trees, gradient boosting, calibrated linear models, and transformer embeddings.
- Conduct fairness/error analysis across genres.
- Add held-out permutation importance for more reliable component analysis.

## 20. Individual reflection

This implementation demonstrates how dimensionality reduction can make high-dimensional NLP features practical for a tree ensemble. TruncatedSVD is fitted inside each training fold, preventing leakage while producing dense latent components. The number of trees, maximum depth, sampled features, and class weighting must be tuned because Random Forest can still overfit imbalanced, limited data.

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
