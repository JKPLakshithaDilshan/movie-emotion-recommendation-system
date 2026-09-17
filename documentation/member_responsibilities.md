# Individual Member Responsibilities

## Project

**Title:** Emotion-Aware Movie Recommendation System Using Machine Learning  
**Module:** IT2011 – Artificial Intelligence and Machine Learning  
**Task:** Seven-class emotion classification and emotion-aware movie recommendation

## Assignment Interpretation

The project has six members. Each member is assigned:

1. One primary preprocessing specialization
2. One individual machine learning model
3. An individual EDA focus
4. Individual hyperparameter tuning and evaluation
5. An individual PDF report and viva presentation

The Progress Review I specification states that every student must handle and present all compulsory preprocessing techniques. Therefore, the primary specialization identifies the technique for which that member provides the group’s most detailed implementation and viva explanation. It does not remove the requirement to demonstrate the other techniques.

## Compulsory Requirements for Every Student

Every student must independently demonstrate the following in their own notebook and viva:

1. Handling missing data
2. Encoding categorical variables
3. Outlier detection and removal or treatment
4. Normalization or scaling
5. Feature engineering
6. Feature selection
7. Dimensionality reduction
8. At least three interpreted EDA visualizations
9. At least one individually assigned machine learning model
10. Multiple tuned varieties of the assigned model
11. Suitable evaluation metrics
12. Train-test validation and cross-validation
13. Model-variety comparison and conclusion
14. Limitations and possible improvements
15. Individual reflection and lessons learned

For each preprocessing technique, every student must provide:

- A clear definition
- Dataset-specific justification
- Executable implementation
- Visible code output
- Interpretation of the result
- Explanation of its effect on subsequent modelling decisions

The shared dataset-preparation script does not replace individual implementation. Every student must execute, explain, justify, and interpret the required techniques in their own notebook.

## Primary Individual Ownership

| Student ID | Student name | Primary preprocessing specialization | Individual model |
|---|---|---|---|
| IT25100112 | Karunarathne H. L. N. A. | Handling missing data | Logistic Regression |
| IT25100083 | Unawatuna H. M. I. D. | Encoding categorical variables | Linear Support Vector Machine |
| IT25100078 | Thujikoshan Y. | Feature selection | Multinomial Naive Bayes |
| IT25100090 | Hettiarachchi H. C. N. G. | Outlier detection and removal | Decision Tree |
| IT25100111 | Chandrasekara C. M. T. D. | Dimensionality reduction | Random Forest |
| IT25100149 | Rajapaksha P. A. D. S. | Normalization and scaling | Multi-Layer Perceptron |

Feature engineering is performed by all six students because it creates the input variables required for every assigned model.

---

## Member 1 – IT25100112

**Name:** Karunarathne H. L. N. A.  
**Primary preprocessing specialization:** Handling missing data  
**Assigned model:** Logistic Regression  
**Notebook:** `IT25100112/IT25100112_logistic_regression.ipynb`

### Primary Specialization Responsibilities

This student is the group’s primary owner of missing-data handling and must explain:

- The difference between null values, blank strings, and unavailable categorical information
- The missing-value audit performed on all raw columns
- Why the raw dataset initially appeared to have no missing values
- How empty genre lists represented by `[]` were discovered
- Why 35 empty genres were encoded as `unknown_genre`
- Why valid movie records were retained instead of deleted
- Missing-value counts before and after preprocessing
- Risks of inappropriate imputation in textual datasets

### Compulsory Preprocessing Demonstrated

The notebook must also show:

- Categorical encoding for emotion and genres
- Rating and review-count outlier analysis
- Numerical-feature scaling
- Description-length feature engineering
- Genre-count feature engineering
- Review-count feature engineering
- TF-IDF text vectorization
- Chi-squared feature selection
- TruncatedSVD dimensionality reduction
- Leakage-safe training and test transformation

### Individual EDA

1. Emotion class-distribution bar chart
2. Mean-rating histogram
3. Mean-rating boxplot by emotion
4. Emotion class-percentage chart

Every visualization must include a written interpretation and its influence on preprocessing or modelling.

### Model Varieties

- Logistic Regression with `C=0.1`
- Logistic Regression with `C=1.0`
- Logistic Regression with `C=10.0`
- Compare no class weighting with `class_weight="balanced"`
- Compare unigram TF-IDF with unigram-bigram TF-IDF
- Compare full features with selected features

### Parameter Tuning

Use `GridSearchCV` or another justified search approach to tune:

- `C`
- `class_weight`
- TF-IDF `ngram_range`
- TF-IDF `min_df`
- Feature-selection size

### Main Evaluation

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Weighted F1-score
- Per-class results
- Confusion matrix
- Five-fold stratified cross-validation
- Training score versus test score
- Model-variety comparison

### Required Outputs

Save outputs under:

```text
IT25100112/results/
```

---

## Member 2 – IT25100083

**Name:** Unawatuna H. M. I. D.  
**Primary preprocessing specialization:** Encoding categorical variables  
**Assigned model:** Linear Support Vector Machine  
**Notebook:** `IT25100083/IT25100083_linear_svm.ipynb`

### Primary Specialization Responsibilities

This student is the group’s primary owner of categorical encoding and must explain:

- Why machine learning algorithms require numerical features
- Target-label encoding for the `emotion` variable
- How encoded class numbers map back to emotion names
- Parsing genre-list strings
- Multi-label genre representation
- Encoding `unknown_genre`
- Why direct movie-name encoding is avoided
- How unseen categories should be handled
- Why encoders must be fitted using training data only

### Compulsory Preprocessing Demonstrated

The notebook must also show:

- Missing and empty-value auditing
- Outlier detection using IQR
- Numerical-feature scaling
- Description word-count engineering
- Description character-count engineering
- Genre-count engineering
- TF-IDF text vectorization
- Chi-squared feature selection
- TruncatedSVD dimensionality reduction
- Leakage-safe pipeline construction

### Individual EDA

1. Description word-count histogram
2. Description word-count boxplot
3. Description length by emotion
4. Description word count versus mean rating

### Model Varieties

- Linear SVM with `C=0.1`
- Linear SVM with `C=0.5`
- Linear SVM with `C=1.0`
- Linear SVM with `C=2.0`
- Compare balanced and unbalanced class weights
- Compare unigram and unigram-bigram features
- Compare full features with selected features

### Parameter Tuning

Tune:

- `C`
- `class_weight`
- TF-IDF `ngram_range`
- TF-IDF `max_features`
- Feature-selection size

### Main Evaluation

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Weighted F1-score
- Per-class results
- Confusion matrix
- Five-fold stratified cross-validation
- Training score versus test score
- Model-variety comparison

### Required Outputs

Save outputs under:

```text
IT25100083/results/
```

---

## Member 3 – IT25100078

**Name:** Thujikoshan Y.  
**Primary preprocessing specialization:** Feature selection  
**Assigned model:** Multinomial Naive Bayes  
**Notebook:** `IT25100078/IT25100078_naive_bayes.ipynb`

### Primary Specialization Responsibilities

This student is the group’s primary owner of feature selection and must explain:

- Why TF-IDF produces high-dimensional features
- The difference between feature engineering and feature selection
- TF-IDF filtering using `min_df`, `max_df`, and `max_features`
- Chi-squared `SelectKBest`
- Why chi-squared selection is suitable for non-negative text features
- How different values of `k` affect performance
- Which feature-selection setting performs best
- Why feature selection must be fitted only on training data

### Compulsory Preprocessing Demonstrated

The notebook must also show:

- Missing-value auditing
- `unknown_genre` handling
- Emotion and genre encoding
- Genre-count and rating outlier analysis
- Non-negative feature normalization
- Genre-token engineering
- Description-length engineering
- TF-IDF text transformation
- TruncatedSVD demonstration
- Leakage-safe preprocessing

Multinomial Naive Bayes requires non-negative input. Therefore, TruncatedSVD must be demonstrated separately and should not be passed directly into the final Multinomial Naive Bayes model if it produces negative component values.

### Individual EDA

1. Top-genre frequency chart
2. Genre-count distribution
3. Genre and emotion heatmap
4. Mean rating by major genre

### Model Varieties

- Multinomial Naive Bayes with `alpha=0.1`
- Multinomial Naive Bayes with `alpha=0.5`
- Multinomial Naive Bayes with `alpha=1.0`
- Multinomial Naive Bayes with `alpha=2.0`
- Compare `fit_prior=True` and `fit_prior=False`
- Compare different feature-selection sizes
- Compare unigram and unigram-bigram TF-IDF

### Parameter Tuning

Tune:

- `alpha`
- `fit_prior`
- TF-IDF `ngram_range`
- TF-IDF `min_df`
- `SelectKBest` value of `k`

### Main Evaluation

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Weighted F1-score
- Per-class results
- Confusion matrix
- Five-fold stratified cross-validation
- Model-variety comparison

### Required Outputs

Save outputs under:

```text
IT25100078/results/
```

---

## Member 4 – IT25100090

**Name:** Hettiarachchi H. C. N. G.  
**Primary preprocessing specialization:** Outlier detection and removal  
**Assigned model:** Decision Tree  
**Notebook:** `IT25100090/IT25100090_decision_tree.ipynb`

### Primary Specialization Responsibilities

This student is the group’s primary owner of outlier handling and must explain:

- The definition of an outlier
- The difference between a valid extreme value and a data error
- Rating-range validation
- IQR-based outlier detection
- Description-length outliers
- Review-count outliers
- Rating-spread outliers
- Before-and-after boxplots
- Number of observations affected
- Removal, capping, or transformation decisions
- Why outlier boundaries affecting modelling must be learned from training data

### Compulsory Preprocessing Demonstrated

The notebook must also show:

- Missing-value auditing
- Unknown-genre handling
- Target and genre encoding
- Numerical-feature scaling for comparison
- Description-length engineering
- Rating-range engineering
- TF-IDF transformation
- Feature selection
- TruncatedSVD dense-feature creation
- Leakage-safe pipeline steps

### Individual EDA

1. Description character-count distribution
2. Description character-count boxplot
3. Engineered numeric-feature correlation matrix
4. Rating-range distribution

### Model Varieties

- Decision Tree using Gini criterion
- Decision Tree using entropy criterion
- `max_depth` values of 10, 20, 40, and unrestricted
- `min_samples_split` values of 2, 5, and 10
- `min_samples_leaf` variations
- Balanced and unbalanced class weights

### Parameter Tuning

Tune:

- `criterion`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `class_weight`

### Main Evaluation

- Training accuracy
- Test accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Weighted F1-score
- Per-class recall
- Confusion matrix
- Cross-validation mean and standard deviation
- Overfitting analysis
- Model-variety comparison

### Required Outputs

Save outputs under:

```text
IT25100090/results/
```

---

## Member 5 – IT25100111

**Name:** Chandrasekara C. M. T. D.  
**Primary preprocessing specialization:** Dimensionality reduction  
**Assigned model:** Random Forest  
**Notebook:** `IT25100111/IT25100111_random_forest.ipynb`

### Primary Specialization Responsibilities

This student is the group’s primary owner of dimensionality reduction and must explain:

- Why TF-IDF creates a sparse high-dimensional matrix
- The difference between feature selection and dimensionality reduction
- Why ordinary PCA is not the main method for sparse TF-IDF data
- Why TruncatedSVD is suitable
- Different SVD component counts
- Explained variance
- Computational benefits
- Information-loss trade-offs
- Comparison of model results using different component counts
- Why SVD must be fitted only using training data

### Compulsory Preprocessing Demonstrated

The notebook must also show:

- Missing-value auditing
- Empty-genre encoding
- Emotion and genre encoding
- Review-count outlier detection
- Log transformation
- Numerical-feature scaling
- Review-popularity feature engineering
- Description-length feature engineering
- TF-IDF transformation
- Feature selection
- Leakage-safe pipeline construction

### Individual EDA

1. Raw review-count distribution
2. Log-transformed review-count distribution
3. Review-count boxplot by emotion
4. Mean rating versus review count
5. TruncatedSVD explained-variance plot

### Model Varieties

- Random Forest with `n_estimators=100`
- Random Forest with `n_estimators=200`
- Random Forest with `n_estimators=300`
- `max_depth` values of 20, 40, and unrestricted
- Different `max_features` settings
- Balanced and unbalanced class weights
- Different SVD component counts

### Parameter Tuning

Tune:

- `n_estimators`
- `max_depth`
- `max_features`
- `min_samples_split`
- `class_weight`
- Number of SVD components

### Main Evaluation

- Training accuracy
- Test accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Weighted F1-score
- Per-class recall
- Confusion matrix
- Cross-validation
- Feature-importance analysis
- Overfitting analysis
- Model-variety comparison

### Required Outputs

Save outputs under:

```text
IT25100111/results/
```

---

## Member 6 – IT25100149

**Name:** Rajapaksha P. A. D. S.  
**Primary preprocessing specialization:** Normalization and scaling  
**Assigned model:** Multi-Layer Perceptron Neural Network  
**Notebook:** `IT25100149/IT25100149_mlp_classifier.ipynb`

### Primary Specialization Responsibilities

This student is the group’s primary owner of normalization and scaling and must explain:

- Why features with different ranges can affect model learning
- The difference between normalization and standardization
- StandardScaler
- MinMaxScaler
- MaxAbsScaler for sparse or non-negative features
- Which scaler is selected for each feature group
- Before-and-after feature statistics
- Why neural networks are sensitive to feature scale
- Why the scaler must be fitted using training data only

### Compulsory Preprocessing Demonstrated

The notebook must also show:

- Missing-value auditing
- Unavailable-genre handling
- Emotion and genre encoding
- Numerical-feature outlier analysis
- Text-length engineering
- Genre-count engineering
- Rating-feature engineering
- Popularity-feature engineering
- TF-IDF transformation
- Feature selection
- TruncatedSVD dense neural-network input
- Leakage-safe preprocessing

### Individual EDA

1. Most frequent description terms
2. Most frequent description bigrams
3. Frequent terms by emotion
4. Scaled-feature before-and-after comparison
5. TruncatedSVD explained-variance plot

### Model Varieties

- Hidden layer `(100,)`
- Hidden layers `(128, 64)`
- ReLU activation
- Tanh activation
- Multiple `alpha` regularization values
- Different initial learning rates
- Early stopping enabled

### Parameter Tuning

Tune:

- `hidden_layer_sizes`
- `activation`
- `alpha`
- `learning_rate_init`
- SVD component count

### Main Evaluation

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Weighted F1-score
- Per-class results
- Confusion matrix
- Cross-validation where computationally practical
- Training-loss curve
- Convergence analysis
- Overfitting analysis
- Model-variety comparison

### Required Outputs

Save outputs under:

```text
IT25100149/results/
```

---

## Shared Data Rules

All students must:

- Use `data/processed/modelling_dataset_7class.csv` for supervised modelling
- Use `data/processed/description_level_dataset.csv` when presenting the complete eight-class EDA
- Use one unique description as one independent observation
- Explain why raw repeated reviews were not used as the primary classifier input
- Explain why `surprise` remains in EDA but is excluded from model training
- Retain `movie_name` for recommendation output rather than direct predictive encoding
- Use random state `42`

## Shared Preprocessing Rules

All students must personally demonstrate:

- Missing-data and blank-string auditing
- Empty-genre handling using `unknown_genre`
- Categorical encoding
- Outlier identification and treatment
- Numerical scaling
- Feature engineering
- Feature selection
- TruncatedSVD dimensionality reduction

Each student must provide deeper evidence for their assigned primary preprocessing specialization.

## Shared Modelling Rules

All students must:

- Use stratified train-test splitting
- Keep the final test set untouched during tuning
- Fit TF-IDF using training data only
- Fit categorical encoders using training data only
- Fit scalers using training data only
- Fit outlier boundaries affecting modelling using training data only
- Fit feature selectors using training data only
- Fit TruncatedSVD using training data only
- Use pipelines where practical
- Use cross-validation on training data
- Compare at least two varieties of the assigned model
- Select the best variety using justified metrics
- Evaluate the selected variety once on the final test set

## Shared Evaluation Rules

Every student must report:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Weighted F1-score
- Per-class metrics
- Confusion matrix
- Cross-validation mean
- Cross-validation standard deviation
- Training-versus-test performance
- Limitations and improvement opportunities

Macro F1 is an important selection metric because the target classes are imbalanced.

## EDA Requirements

Each student must present at least three visualizations. Every visualization must contain:

- A meaningful title
- Labelled axes
- A readable legend where applicable
- Saved PNG or JPEG output
- A written interpretation
- An explanation of how the result affected preprocessing or modelling

Each student’s images must be saved under:

```text
STUDENT_ID/results/eda_visualizations/
```

## Model Output Requirements

Each student must save:

- Hyperparameter-search results
- Best parameter combination
- Cross-validation results
- Classification report
- Confusion matrix
- Model-variety comparison
- Final conclusions

Outputs must be saved under:

```text
STUDENT_ID/results/outputs/
STUDENT_ID/results/model_visualizations/
STUDENT_ID/results/logs/
```

## Individual Completion Rule

A student’s contribution is complete only when their notebook contains:

- All compulsory preprocessing techniques
- Detailed evidence for their primary preprocessing specialization
- At least three interpreted EDA visualizations
- Their assigned machine learning model
- Multiple tuned model varieties
- Appropriate validation and evaluation
- Individual conclusions
- Individual limitations
- Individual reflection

The final project therefore contains:

- Six primary preprocessing specializations
- Six individual preprocessing implementations
- Six individual EDA sections
- Six model-training implementations
- Six hyperparameter-tuning sections
- Six individual evaluations
- Six individual PDF reports
- One final group model comparison
- One final group report

## Report Responsibility

Every student must prepare an individual report named using their Student ID.

Each report must include:

1. Introduction and Problem Statement
2. Dataset Description
3. Data Preprocessing and EDA
4. Model Design and Implementation
5. Model Evaluation and Comparison
6. Ethical Considerations and Bias Mitigation
7. AI Tool Usage Declaration and Transparency
8. Individual Reflection and Lessons Learned
9. References

A separate final group report will combine the shared problem, preprocessing decisions, six-model comparison, best-model selection, recommendation component, ethics, AI declaration, and overall conclusions.

## AI Tool Transparency

Every member must truthfully declare:

- AI tool name and version, if known
- Tasks supported by the AI tool
- Extent of use
- Verification and testing performed
- Modifications made to AI-assisted output

Every member remains responsible for understanding and explaining all submitted work.