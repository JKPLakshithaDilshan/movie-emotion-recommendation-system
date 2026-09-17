from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if not (PROJECT_ROOT / "data").exists():
    raise FileNotFoundError("Run this script from inside the repository.")

DATA_PATH = PROJECT_ROOT / "data/processed/modelling_dataset_7class.csv"
FULL_DATA_PATH = PROJECT_ROOT / "data/processed/description_level_dataset.csv"
GROUP_OUTPUT = PROJECT_ROOT / "final_group_results/outputs"
GROUP_VISUALS = PROJECT_ROOT / "final_group_results/visualizations"
GROUP_REPORT_DIR = PROJECT_ROOT / "final_group_report"
GROUP_REPORT_DIR.mkdir(parents=True, exist_ok=True)

MAX_PAGES = 15

MEMBERS = [
    {
        "id": "IT25100112",
        "name": "Karunarathne H. L. N. A.",
        "model": "Logistic Regression",
        "primary": "Handling missing data",
        "prefix": "logistic_regression",
        "notebook": "IT25100112_logistic_regression.ipynb",
        "model_reason": "A regularized linear classifier is efficient and interpretable for high-dimensional TF-IDF features.",
        "tuning": "C = 0.1, 1.0, 10.0; balanced/unbalanced class weights; unigram and unigram-bigram TF-IDF; chi-squared feature counts.",
        "eda": [
            ("01_emotion_class_distribution.png", "The target is severely imbalanced, with sadness dominant and disgust the smallest modelled class."),
            ("02_mean_rating_histogram.png", "Mean ratings are concentrated in the upper-middle range and do not independently separate emotions."),
            ("03_mean_rating_boxplot_by_emotion.png", "Rating distributions overlap across emotions; rating is retained only as supporting information."),
            ("04_emotion_class_percentage.png", "Percentage shares confirm the need for macro F1 and class weighting during tuning."),
        ],
    },
    {
        "id": "IT25100083",
        "name": "Unawatuna H. M. I. D.",
        "model": "Linear Support Vector Machine",
        "primary": "Encoding categorical variables",
        "prefix": "linear_svm",
        "notebook": "IT25100083_linear_svm.ipynb",
        "model_reason": "Linear SVM is well suited to sparse high-dimensional text and maximizes the class-separating margin.",
        "tuning": "C = 0.1, 0.5, 1.0, 2.0; balanced/unbalanced class weights; unigram and unigram-bigram TF-IDF; feature-selection sizes.",
        "eda": [
            ("01_description_word_count_histogram.png", "Description length is right-skewed, motivating outlier treatment and scaled engineered features."),
            ("02_description_word_count_boxplot.png", "Long descriptions are valid observations; they are clipped rather than deleted."),
            ("03_description_length_by_emotion.png", "Length distributions overlap strongly, so semantic TF-IDF features remain essential."),
            ("04_word_count_vs_mean_rating.png", "Word count and rating show weak separation and function only as supporting signals."),
        ],
    },
    {
        "id": "IT25100078",
        "name": "Thujikoshan Y.",
        "model": "Multinomial Naive Bayes",
        "primary": "Feature selection",
        "prefix": "naive_bayes",
        "notebook": "IT25100078_naive_bayes.ipynb",
        "model_reason": "Multinomial Naive Bayes is an efficient probabilistic baseline for non-negative TF-IDF and encoded features.",
        "tuning": "Alpha = 0.1, 0.5, 1.0, 2.0; fitted/uniform priors; 500, 1,000, and 1,500 selected features; unigram/bigram TF-IDF.",
        "eda": [
            ("01_top_genre_frequency.png", "A few genres dominate, while rare genres offer limited class evidence."),
            ("02_genre_count_distribution.png", "Most descriptions contain a small number of genres; unavailable genres use an explicit category."),
            ("03_genre_emotion_heatmap.png", "Genre-emotion frequencies vary but are also influenced by overall class imbalance."),
            ("04_mean_rating_by_major_genre.png", "Ratings differ modestly by genre and are insufficient as standalone predictors."),
        ],
    },
    {
        "id": "IT25100090",
        "name": "Hettiarachchi H. C. N. G.",
        "model": "Decision Tree",
        "primary": "Outlier detection and treatment",
        "prefix": "decision_tree",
        "notebook": "IT25100090_decision_tree.ipynb",
        "model_reason": "Decision Trees provide nonlinear rules and create a useful contrast with linear text classifiers.",
        "tuning": "Gini/entropy; maximum depth 10, 20, 40, unrestricted; minimum split 2, 5, 10; balanced/unbalanced class weights.",
        "eda": [
            ("01_character_count_distribution.png", "Character counts are right-skewed and include legitimate long descriptions."),
            ("02_character_count_boxplot.png", "IQR analysis identifies extremes that are winsorized to preserve minority observations."),
            ("03_numeric_feature_correlation.png", "Length, rating, and popularity summaries contain redundancy, supporting feature reduction."),
            ("04_rating_range_distribution.png", "Most descriptions have limited rating spread, with a tail of more variable movies."),
        ],
    },
    {
        "id": "IT25100111",
        "name": "Chandrasekara C. M. T. D.",
        "model": "Random Forest",
        "primary": "Dimensionality reduction",
        "prefix": "random_forest",
        "notebook": "IT25100111_random_forest.ipynb",
        "model_reason": "Random Forest aggregates many trees to reduce variance and provides an ensemble comparison with a single tree.",
        "tuning": "100, 200, 300 trees; maximum depth 20, 40, unrestricted; sqrt/log2 feature sampling; balanced/unbalanced class weights.",
        "eda": [
            ("01_raw_review_count_distribution.png", "Raw popularity is heavily right-skewed and requires transformation."),
            ("02_log_review_count_distribution.png", "Log1p compression reduces the extreme popularity tail while preserving order."),
            ("03_review_count_by_emotion.png", "Popularity overlaps across emotions and cannot determine the label alone."),
            ("04_mean_rating_vs_review_count.png", "Rating and popularity provide supporting rather than decisive emotion signals."),
        ],
    },
    {
        "id": "IT25100149",
        "name": "Rajapaksha P. A. D. S.",
        "model": "Multi-Layer Perceptron",
        "primary": "Normalization and scaling",
        "prefix": "mlp",
        "notebook": "IT25100149_mlp_classifier.ipynb",
        "model_reason": "An MLP can learn nonlinear relationships from dense scaled SVD components and offers a neural-network comparison.",
        "tuning": "Hidden layers (100,) and (128, 64); ReLU/tanh; alpha 0.0001/0.001; learning rates 0.001/0.01; early stopping.",
        "eda": [
            ("01_most_frequent_terms.png", "Frequent corpus terms appear across classes and are down-weighted by TF-IDF."),
            ("02_most_frequent_bigrams.png", "Bigrams retain short context but occur less frequently than individual words."),
            ("03_frequent_terms_by_emotion.png", "Emotion vocabularies show some differences alongside substantial overlap."),
            ("04_svd_explained_variance.png", "SVD converts sparse features into a manageable dense representation for the MLP."),
        ],
    },
]


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="ReportTitle",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=23,
    leading=28,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#123B5D"),
    spaceAfter=18,
))
styles.add(ParagraphStyle(
    name="Section",
    parent=styles["Heading1"],
    fontName="Helvetica-Bold",
    fontSize=15,
    leading=18,
    textColor=colors.HexColor("#123B5D"),
    spaceBefore=8,
    spaceAfter=8,
))
styles.add(ParagraphStyle(
    name="Subsection",
    parent=styles["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=11.5,
    leading=14,
    textColor=colors.HexColor("#176B87"),
    spaceBefore=6,
    spaceAfter=5,
))
styles.add(ParagraphStyle(
    name="BodySmall",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=9.2,
    leading=12.2,
    spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="CoverMeta",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=11,
    leading=17,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#334155"),
))
styles.add(ParagraphStyle(
    name="Caption",
    parent=styles["BodyText"],
    fontName="Helvetica-Oblique",
    fontSize=8.2,
    leading=10.5,
    textColor=colors.HexColor("#475569"),
    spaceAfter=8,
))


def footer(canvas, document):
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(1.7 * cm, 1.35 * cm, width - 1.7 * cm, 1.35 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(1.7 * cm, 0.9 * cm, "IT2011 - Artificial Intelligence and Machine Learning")
    canvas.drawRightString(width - 1.7 * cm, 0.9 * cm, f"Page {document.page}")
    canvas.restoreState()


def paragraph(text, style="BodySmall"):
    return Paragraph(text, styles[style])


def styled_table(data, widths=None, font_size=7.5, header=True):
    body_style = ParagraphStyle(
        name=f"TableBody{font_size}",
        fontName="Helvetica",
        fontSize=font_size,
        leading=font_size + 2,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=0,
        splitLongWords=True,
    )

    header_style = ParagraphStyle(
        name=f"TableHeader{font_size}",
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )

    wrapped_data = []

    for row_index, row in enumerate(data):
        cell_style = (
            header_style
            if header and row_index == 0
            else body_style
        )

        wrapped_data.append([
            Paragraph(
                escape(str(value)).replace("\n", "<br/>"),
                cell_style,
            )
            for value in row
        ])

    table = Table(
        wrapped_data,
        colWidths=widths,
        repeatRows=1 if header else 0,
        hAlign="LEFT",
    )

    commands = [
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F8FAFC")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]

    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#123B5D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ])

    table.setStyle(TableStyle(commands))
    return table

def scaled_image(path: Path, max_width=16.5 * cm, max_height=8.6 * cm):
    if not path.exists():
        raise FileNotFoundError(f"Required report image missing: {path}")
    image = Image(str(path))
    ratio = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * ratio
    image.drawHeight = image.imageHeight * ratio
    return image


def metrics_for(member):
    path = (
        PROJECT_ROOT / member["id"] / "results/outputs"
        / f"{member['prefix']}_metrics.csv"
    )
    return pd.read_csv(path, index_col=0)["score"]


def summary_for(member):
    path = (
        PROJECT_ROOT / member["id"] / "results/outputs"
        / f"{member['prefix']}_summary.json"
    )
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def add_cover(story, title, subtitle, identity_lines):
    story.extend([
        Spacer(1, 2.3 * cm),
        paragraph("SLIIT Faculty of Computing", "CoverMeta"),
        paragraph("Year 2 Semester 1 - 2026", "CoverMeta"),
        Spacer(1, 1.0 * cm),
        Paragraph(title, styles["ReportTitle"]),
        paragraph(subtitle, "CoverMeta"),
        Spacer(1, 1.2 * cm),
    ])
    for line in identity_lines:
        story.append(paragraph(line, "CoverMeta"))
    story.extend([
        Spacer(1, 1.5 * cm),
        paragraph("Emotion-Aware Movie Recommendation System", "CoverMeta"),
        paragraph("Final Report", "CoverMeta"),
        PageBreak(),
    ])


def build_individual_report(member, full_df, model_df, group_comparison):
    output_dir = PROJECT_ROOT / member["id"] / "report"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{member['id']}.pdf"

    metrics = metrics_for(member)
    summary = summary_for(member)
    confusion = (
        PROJECT_ROOT / member["id"] / "results/model_visualizations"
        / f"{member['prefix']}_confusion_matrix.png"
    )
    story = []
    add_cover(
        story,
        "Individual Machine Learning Report",
        "IT2011 - Artificial Intelligence and Machine Learning",
        [
            f"<b>Student:</b> {member['name']}",
            f"<b>Student ID:</b> {member['id']}",
            f"<b>Assigned model:</b> {member['model']}",
            f"<b>Primary preprocessing specialization:</b> {member['primary']}",
        ],
    )

    story.extend([
        paragraph("Executive Summary", "Section"),
        paragraph(
            f"This report presents an individual implementation of {member['model']} for seven-class emotion prediction from movie descriptions. The work independently demonstrates missing-data handling, categorical encoding, outlier treatment, normalization/scaling, feature engineering, feature selection, dimensionality reduction, exploratory analysis, tuning, cross-validation, and held-out evaluation. The achieved test macro F1 was <b>{metrics['macro_f1']:.4f}</b> and test accuracy was <b>{metrics['test_accuracy']:.4f}</b>.",
        ),
        paragraph("1. Introduction and Problem Statement", "Section"),
        paragraph(
            "Conventional movie recommenders emphasize popularity, ratings, or viewing history and may not represent the emotional character of movie content. This project predicts one of seven emotions - anger, anticipation, disgust, fear, joy, optimism, or sadness - from a movie description and combines the result with genre and rating information for emotion-aware recommendation.",
        ),
        paragraph("2. Dataset Description", "Section"),
        paragraph(
            f"The assigned IMDb Movie Reviews - Genres, Descriptions and Emotions CSV originally contained 46,173 rows and eight columns. Description-level aggregation produced {len(full_df):,} independent descriptions for EDA and {len(model_df):,} records for seven-class modelling. The surprise class was retained for EDA but excluded from training because only three independent examples were available.",
        ),
        styled_table([
            ["Characteristic", "Value"],
            ["Raw observations", "46,173"],
            ["Full EDA observations", f"{len(full_df):,}"],
            ["Seven-class observations", f"{len(model_df):,}"],
            ["Target", "emotion"],
            ["Primary text feature", "Description"],
            ["Supporting features", "Genre, rating, length, popularity"],
            ["Missing processed values", str(int(model_df.isna().sum().sum()))],
        ], widths=[5.2 * cm, 11.0 * cm]),
        PageBreak(),
    ])

    story.extend([
        paragraph("3. Data Preprocessing and Exploratory Data Analysis", "Section"),
        paragraph(
            f"The primary specialization was <b>{member['primary']}</b>. The complete workflow still independently demonstrated every compulsory preprocessing technique. All learned transformations used during modelling were placed inside a pipeline and fitted on training folds only.",
        ),
        styled_table([
            ["Technique", "Implementation and justification"],
            ["Missing data", "Audited nulls and blanks; represented unavailable genre information as unknown_genre; used fold-fitted imputers."],
            ["Categorical encoding", "One-hot encoded genre categories with unknown-category handling; demonstrated target label encoding."],
            ["Outliers", "Detected numeric extremes using the 1.5 IQR rule and clipped them instead of deleting valid minority observations."],
            ["Scaling", "Scaled numeric inputs; preserved non-negative values before chi-squared selection and standardized dense neural input where applicable."],
            ["Feature engineering", "Created rating spread, text length, genre count, review popularity, and log-count features."],
            ["Feature selection", "Used training-fold chi-squared selection to retain target-associated non-negative features."],
            ["Dimensionality reduction", "Demonstrated TruncatedSVD; used it in dense-input models where appropriate."],
        ], widths=[3.8 * cm, 12.4 * cm], font_size=7.2),
        Spacer(1, 0.2 * cm),
        paragraph("EDA Visualizations and Interpretation", "Subsection"),
    ])

    eda_dir = PROJECT_ROOT / member["id"] / "results/eda_visualizations"
    for index, (filename, interpretation) in enumerate(member["eda"], start=1):
        story.append(KeepTogether([
            scaled_image(eda_dir / filename, max_height=7.0 * cm),
            paragraph(f"Figure {index}. {interpretation}", "Caption"),
        ]))
        if index == 2:
            story.append(PageBreak())
    story.append(PageBreak())

    story.extend([
        paragraph("4. Model Design and Implementation", "Section"),
        paragraph(f"<b>Suitability.</b> {member['model_reason']}"),
        paragraph(
            "The implementation used pandas, NumPy, Matplotlib, Seaborn, and scikit-learn. A stratified 80:20 split with random state 42 preserved class proportions. GridSearchCV used five-fold StratifiedKFold and optimized macro F1. The final test set remained untouched during tuning.",
        ),
        paragraph("Hyperparameter varieties", "Subsection"),
        paragraph(member["tuning"]),
        paragraph("Selected parameters", "Subsection"),
        paragraph(str(summary["best_parameters"]).replace("{", "").replace("}", "")),
        paragraph("5. Model Evaluation and Comparison", "Section"),
        styled_table([
            ["Metric", "Score"],
            ["Training accuracy", f"{metrics['train_accuracy']:.4f}"],
            ["Test accuracy", f"{metrics['test_accuracy']:.4f}"],
            ["Macro precision", f"{metrics['macro_precision']:.4f}"],
            ["Macro recall", f"{metrics['macro_recall']:.4f}"],
            ["Macro F1", f"{metrics['macro_f1']:.4f}"],
            ["Weighted F1", f"{metrics['weighted_f1']:.4f}"],
            ["Best CV macro F1", f"{metrics['best_cv_macro_f1']:.4f}"],
        ], widths=[7.0 * cm, 4.0 * cm]),
        Spacer(1, 0.25 * cm),
        paragraph(
            "Macro F1 was emphasized because it gives equal importance to every class. Weighted F1 and accuracy are also reported but are influenced more strongly by the dominant sadness and joy classes.",
        ),
        scaled_image(confusion, max_height=8.0 * cm),
        paragraph("Confusion matrix for the selected tuned variety.", "Caption"),
        PageBreak(),
    ])

    compact_comparison = group_comparison[[
        "rank", "model", "test_accuracy", "macro_f1", "weighted_f1", "cv_macro_f1"
    ]].copy()
    comparison_data = [["Rank", "Model", "Accuracy", "Macro F1", "Weighted F1", "CV Macro F1"]]
    for _, row in compact_comparison.iterrows():
        comparison_data.append([
            str(int(row["rank"])), row["model"], f"{row['test_accuracy']:.4f}",
            f"{row['macro_f1']:.4f}", f"{row['weighted_f1']:.4f}", f"{row['cv_macro_f1']:.4f}",
        ])
    gap = metrics["train_accuracy"] - metrics["test_accuracy"]
    story.extend([
        paragraph("Group-level comparison", "Subsection"),
        styled_table(comparison_data, widths=[1.1 * cm, 4.4 * cm, 2.4 * cm, 2.4 * cm, 2.5 * cm, 2.5 * cm], font_size=6.8),
        Spacer(1, 0.25 * cm),
        paragraph(
            f"For this model, the training-test accuracy gap was {gap:.4f}. A large gap indicates overfitting; similarly weak training and test performance may indicate underfitting. Class-level results showed that disgust remained especially difficult because only 73 independent examples were available.",
        ),
        paragraph("6. Ethical Considerations and Bias Mitigation", "Section"),
        paragraph(
            "The class imbalance creates unequal error risks and may underrepresent minority emotions. Emotion labels are subjective, descriptions may contain cultural or linguistic bias, and a low-confidence prediction should not be treated as a psychological assessment. Mitigation included macro-F1 reporting, stratification, balanced-class tuning where supported, class-level evaluation, transparent limitations, and retaining no personal user information. A production system should add confidence calibration, abstention, user feedback, and fairness monitoring.",
        ),
        paragraph("7. AI Tool Usage Declaration and Transparency", "Section"),
        styled_table([
            ["Tool", "Use", "Extent", "Verification"],
            ["ChatGPT / Codex", "Code structure, debugging suggestions, documentation organization, explanation and grammar refinement", "Moderate", "All notebooks were executed; metrics, artifacts, model reloads, and outputs were checked. The student remains responsible for understanding and explaining every cell."],
        ], widths=[3.0 * cm, 5.0 * cm, 2.0 * cm, 6.2 * cm], font_size=6.9),
        paragraph("8. Individual Reflections and Lessons Learned", "Section"),
        paragraph(
            f"The work showed that preprocessing must be justified by the data and kept inside training folds. The primary responsibility, {member['primary'].lower()}, was not an isolated step: it interacted with feature representation, computational cost, and generalization. The final results also showed why macro F1, class-level recall, and overfitting analysis are necessary instead of relying only on accuracy.",
        ),
        paragraph("9. References", "Section"),
        paragraph(
            "1. Fahad Rehman. IMDb Movie Reviews - Genres, Descriptions and Emotions. Kaggle.<br/>"
            "2. Pedregosa et al. Scikit-learn: Machine Learning in Python. JMLR, 2011.<br/>"
            "3. Salton and Buckley. Term-weighting approaches in automatic text retrieval, 1988.<br/>"
            "4. SLIIT IT2011 Assignment Specification and Final Evaluation Guidelines, 2026.<br/>"
            "5. Project notebooks, dataset audit, data dictionary, and executed result files.",
        ),
    ])

    document = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        rightMargin=1.7 * cm, leftMargin=1.7 * cm,
        topMargin=1.55 * cm, bottomMargin=1.65 * cm,
        title=f"{member['id']} Individual Report",
        author=member["name"],
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    pages = len(PdfReader(str(output_path)).pages)
    if pages > MAX_PAGES:
        raise ValueError(f"{output_path} has {pages} pages; maximum is {MAX_PAGES}.")
    return output_path, pages


def build_group_report(full_df, model_df, comparison):
    output_path = GROUP_REPORT_DIR / "IT2011_Group_Report.pdf"
    story = []
    add_cover(
        story,
        "Final Group Report",
        "IT2011 - Artificial Intelligence and Machine Learning",
        [
            "<b>Project:</b> Emotion-Aware Movie Recommendation System",
            "<b>Task:</b> Seven-class emotion classification and recommendation",
            "<b>Academic year:</b> Year 2 Semester 1 - 2026",
        ],
    )

    member_rows = [["Student ID", "Student", "Assigned model"]]
    for member in MEMBERS:
        member_rows.append([member["id"], member["name"], member["model"]])

    story.extend([
        paragraph("Executive Summary", "Section"),
        paragraph(
            "This project developed an end-to-end NLP classification and recommendation workflow using the assigned movie dataset. Six independently tuned model families were evaluated on a common seven-class description-level dataset. Linear SVM achieved the highest held-out macro F1 (0.2523) and accuracy (0.3641) and was selected for the recommendation demonstration. Performance remains limited by class imbalance, subjective labels, and a small number of independent descriptions.",
        ),
        styled_table(member_rows, widths=[3.0 * cm, 7.3 * cm, 5.9 * cm], font_size=7.2),
        paragraph("1. Introduction and Problem Statement", "Section"),
        paragraph(
            "Users may wish to find movies that match their current mood, while conventional recommenders focus on popularity, ratings, and behavior. The proposed solution predicts emotion from a movie description and combines predicted emotion with genre, rating, and popularity signals to rank catalogue recommendations.",
        ),
        paragraph("2. Dataset Description", "Section"),
        paragraph(
            f"The raw CSV contained 46,173 rows and eight columns. Auditing identified repeated reviews with inconsistent targets, so one unique, label-consistent movie description became the modelling unit. The processed full EDA dataset contains {len(full_df):,} descriptions and eight emotions. The modelling dataset contains {len(model_df):,} descriptions and seven classes after excluding three surprise observations.",
        ),
        PageBreak(),
        paragraph("3. Preprocessing and EDA", "Section"),
        styled_table([
            ["Stage", "Group implementation"],
            ["Quality audit", "Validated schema, nulls, blanks, rating range, exact duplicates, description uniqueness, and conflicting labels."],
            ["Missing genres", "Encoded 35 empty genre lists using the explicit unknown_genre category."],
            ["Aggregation", "Created one independent record per description and aggregated rating and review statistics."],
            ["Feature engineering", "Text lengths, genre count, rating spread, review counts, unique counts, and log popularity."],
            ["Encoding", "TF-IDF for descriptions and one-hot encoding for genres."],
            ["Outliers and scaling", "Fold-fitted IQR clipping plus appropriate numerical scaling."],
            ["Selection and reduction", "Chi-squared feature selection and TruncatedSVD where dense inputs were needed."],
            ["Leakage control", "All learned transformations were fitted inside pipelines on training folds only."],
        ], widths=[4.0 * cm, 12.2 * cm], font_size=7.2),
        paragraph(
            "EDA showed severe target imbalance, overlapping rating/length distributions, skewed popularity, genre imbalance, and vocabulary overlap. These findings motivated stratification, macro-F1 optimization, log transformation, class-aware tuning, and text-focused modelling.",
        ),
        paragraph("4. Model Design and Implementation", "Section"),
        styled_table([
            ["Model", "Core rationale", "Main tuned parameters"],
            ["Logistic Regression", "Sparse linear baseline", "C, class weight, n-grams, selected features"],
            ["Linear SVM", "Maximum-margin sparse text classifier", "C, class weight, n-grams, selected features"],
            ["Multinomial NB", "Non-negative probabilistic text baseline", "alpha, priors, feature count, n-grams"],
            ["Decision Tree", "Nonlinear rule model", "criterion, depth, split, class weight"],
            ["Random Forest", "Bagged tree ensemble", "tree count, depth, sampled features, class weight"],
            ["MLP", "Dense nonlinear neural model", "architecture, activation, alpha, learning rate"],
        ], widths=[3.0 * cm, 6.0 * cm, 7.2 * cm], font_size=6.8),
        paragraph(
            "Every model used an 80:20 stratified split, random state 42, five-fold stratified cross-validation, GridSearchCV, and macro F1 as the tuning objective. The held-out test set was evaluated after tuning.",
        ),
        PageBreak(),
    ])

    comparison_data = [["Rank", "Model", "Accuracy", "Macro F1", "Weighted F1", "CV Macro F1", "Gap"]]
    for _, row in comparison.iterrows():
        comparison_data.append([
            str(int(row["rank"])), row["model"], f"{row['test_accuracy']:.4f}",
            f"{row['macro_f1']:.4f}", f"{row['weighted_f1']:.4f}",
            f"{row['cv_macro_f1']:.4f}", f"{row['accuracy_gap']:.4f}",
        ])

    story.extend([
        paragraph("5. Evaluation and Comparison", "Section"),
        styled_table(comparison_data, widths=[0.9 * cm, 4.0 * cm, 2.1 * cm, 2.1 * cm, 2.2 * cm, 2.3 * cm, 1.6 * cm], font_size=6.5),
        Spacer(1, 0.2 * cm),
        scaled_image(GROUP_VISUALS / "final_model_comparison.png", max_height=8.0 * cm),
        paragraph("Final held-out macro-F1 and accuracy comparison across six model families.", "Caption"),
        paragraph(
            "Linear SVM was selected because it produced the highest held-out macro F1 and accuracy. Logistic Regression and Naive Bayes were competitive, while the tree-based models strongly overfit dense SVD components. MLP achieved high accuracy relative to several models but weaker macro F1, showing bias toward larger classes. Disgust remained the weakest class across models.",
        ),
        paragraph("Recommendation Demonstration", "Subsection"),
        paragraph(
            "The saved Linear SVM pipeline predicts an emotion from a supplied description. Catalogue items are filtered by predicted emotion, optional genre, and minimum rating, then ranked using 75% normalized rating and 25% log-popularity. The executed demonstration predicted sadness and produced ranked drama recommendations. This is a content-based academic prototype rather than a production personalization service.",
        ),
        paragraph("6. Ethical Considerations and Bias Mitigation", "Section"),
        paragraph(
            "Severe class imbalance, subjective emotion annotation, possible cultural bias, and uneven genre representation can create unequal performance. False predictions may frustrate users and should never be treated as psychological assessments. Mitigation included stratification, macro-F1 reporting, balanced weights where supported, class-level metrics, transparent limitations, and no use of personal data. Future deployment should calibrate confidence, allow abstention and feedback, and monitor subgroup errors.",
        ),
        PageBreak(),
        paragraph("7. AI Tool Usage Declaration and Transparency", "Section"),
        styled_table([
            ["Tool", "Support provided", "Extent", "Group verification"],
            ["ChatGPT / Codex", "Problem framing, code structuring, debugging suggestions, visualization ideas, tuning suggestions, documentation and language refinement", "Moderate", "All scripts and notebooks were executed. Metrics were read from generated files, models were reloaded, outputs were inspected, and students retain responsibility for understanding their work."],
        ], widths=[3.0 * cm, 5.7 * cm, 2.0 * cm, 5.5 * cm], font_size=6.9),
        paragraph("8. Reflections and Lessons Learned", "Section"),
        paragraph(
            "The group learned that the modelling unit is as important as the algorithm: conflicting repeated reviews required description-level aggregation. Leakage-safe pipelines were essential because TF-IDF, imputers, scalers, feature selectors, and SVD learn from data. Macro F1 exposed weaknesses hidden by accuracy. Simple sparse linear models generalized better than more complex dense models on the limited dataset, demonstrating that complexity does not guarantee performance.",
        ),
        paragraph("9. References", "Section"),
        paragraph(
            "1. Fahad Rehman. IMDb Movie Reviews - Genres, Descriptions and Emotions. Kaggle.<br/>"
            "2. Pedregosa et al. Scikit-learn: Machine Learning in Python. JMLR, 2011.<br/>"
            "3. Salton and Buckley. Term-weighting approaches in automatic text retrieval, 1988.<br/>"
            "4. SLIIT IT2011 Assignment Specification, Final Evaluation - Documentation, and Final Evaluation - Implementation, 2026.<br/>"
            "5. Project dataset audit, data dictionary, member responsibilities, executed notebooks, and generated result files.",
        ),
    ])

    document = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        rightMargin=1.7 * cm, leftMargin=1.7 * cm,
        topMargin=1.55 * cm, bottomMargin=1.65 * cm,
        title="IT2011 Final Group Report",
        author="Emotion-Aware Movie Recommendation System Group",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    pages = len(PdfReader(str(output_path)).pages)
    if pages > MAX_PAGES:
        raise ValueError(f"{output_path} has {pages} pages; maximum is {MAX_PAGES}.")
    return output_path, pages


def main():
    full_df = pd.read_csv(FULL_DATA_PATH)
    model_df = pd.read_csv(DATA_PATH)
    comparison = pd.read_csv(GROUP_OUTPUT / "final_six_model_comparison.csv")

    generated = []
    for member in MEMBERS:
        generated.append(
            build_individual_report(member, full_df, model_df, comparison)
        )
    generated.append(build_group_report(full_df, model_df, comparison))

    print("Generated and page-validated reports:")
    for path, pages in generated:
        print(f"- {path.relative_to(PROJECT_ROOT)}: {pages} pages")


if __name__ == "__main__":
    main()
