
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, classification_report


# ==========================================================
# SETTINGS
# ==========================================================

DATASET = "job_interview.csv"
MODEL_FILE = "interview_model.pkl"

RANDOM_STATE = 42


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

print("=" * 70)
print("        AI INTERVIEW ANALYZER - IMPROVED MODEL")
print("=" * 70)

df = pd.read_csv(DATASET)

print("\nDataset loaded successfully!")

print("Rows    :", len(df))
print("Columns :", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())


# ==========================================================
# 2. REQUIRED COLUMNS
# ==========================================================

required_columns = [
    "Transcript",
    "Resume",
    "Job_Description",
    "decision"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# ==========================================================
# 3. SELECT COLUMNS
# ==========================================================

df = df[required_columns].copy()


# ==========================================================
# 4. HANDLE MISSING VALUES
# ==========================================================

for column in [
    "Transcript",
    "Resume",
    "Job_Description"
]:

    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
    )


df["decision"] = (
    df["decision"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)


# ==========================================================
# 5. KEEP ONLY VALID TARGETS
# ==========================================================

df = df[
    df["decision"].isin(
        ["select", "reject"]
    )
].copy()


if len(df) == 0:

    raise ValueError(
        "No valid SELECT / REJECT records found."
    )


print("\nDecision distribution:")
print(df["decision"].value_counts())


# ==========================================================
# 6. CREATE SEPARATE TEXT FEATURES
# ==========================================================

# We give each field its own TF-IDF vectorizer.
#
# Resume          → candidate skills/experience
# Transcript      → interview responses
# Job Description → required skills
#
# This is better than treating everything as one plain text field.

features = FeatureUnion([

    (
        "resume",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            sublinear_tf=True
        )
    ),

    (
        "transcript",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            sublinear_tf=True
        )
    ),

    (
        "job",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            sublinear_tf=True
        )
    )
])


# ==========================================================
# 7. PREPARE TEXT
# ==========================================================

# FeatureUnion expects one text input.
# We therefore create clearly separated sections.

df["model_text"] = (
    " RESUME_SECTION "
    + df["Resume"]
    + " TRANSCRIPT_SECTION "
    + df["Transcript"]
    + " JOB_SECTION "
    + df["Job_Description"]
)


X = df["model_text"]
y = df["decision"]


# ==========================================================
# 8. TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y
)


print("\nTraining records:", len(X_train))
print("Testing records :", len(X_test))


# ==========================================================
# 9. TRANSFORM TEXT
# ==========================================================

print("\nCreating TF-IDF features...")

X_train_tfidf = features.fit_transform(
    X_train
)

X_test_tfidf = features.transform(
    X_test
)

print(
    "TF-IDF feature matrix:",
    X_train_tfidf.shape
)


# ==========================================================
# 10. TRY MULTIPLE CLASSIFIERS
# ==========================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=3000,
            C=2.0,
            class_weight="balanced"
        ),

    "Linear SVM":
        LinearSVC(
            C=1.0,
            class_weight="balanced"
        ),

    "Complement Naive Bayes":
        ComplementNB(
            alpha=0.1
        )
}


results = {}


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)


# ==========================================================
# 11. TRAIN EACH MODEL
# ==========================================================

for name, classifier in models.items():

    print(
        f"\nTraining: {name}"
    )

    classifier.fit(
        X_train_tfidf,
        y_train
    )

    predictions = classifier.predict(
        X_test_tfidf
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results[name] = {
        "model": classifier,
        "accuracy": accuracy
    }

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )


# ==========================================================
# 12. DISPLAY ALL RESULTS
# ==========================================================

print("\n" + "=" * 70)
print("ALL MODEL ACCURACIES")
print("=" * 70)

for name, result in results.items():

    print(
        f"{name:<30} "
        f"{result['accuracy'] * 100:.2f}%"
    )


# ==========================================================
# 13. FIND BEST MODEL
# ==========================================================

best_name = max(
    results,
    key=lambda x: results[x]["accuracy"]
)

best_model = results[
    best_name
]["model"]

best_accuracy = results[
    best_name
]["accuracy"]


print("\n" + "=" * 70)

print(
    "BEST MODEL:",
    best_name
)

print(
    f"BEST ACCURACY: "
    f"{best_accuracy * 100:.2f}%"
)

print("=" * 70)


# ==========================================================
# 14. BEST MODEL REPORT
# ==========================================================

best_predictions = best_model.predict(
    X_test_tfidf
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        best_predictions
    )
)


# ==========================================================
# 15. SAVE EVERYTHING REQUIRED BY APP
# ==========================================================

model_package = {

    "vectorizer": features,

    "model": best_model,

    "model_name": best_name,

    "accuracy": best_accuracy

}


joblib.dump(
    model_package,
    MODEL_FILE
)


print("\n" + "=" * 70)

print(
    "MODEL SAVED SUCCESSFULLY!"
)

print(
    "File:",
    MODEL_FILE
)

print(
    f"Saved Model Accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print("=" * 70)


# ==========================================================
# 16. SAMPLE PREDICTION
# ==========================================================

sample_text = X_test.iloc[0]

sample_actual = y_test.iloc[0]

sample_features = features.transform(
    [sample_text]
)

sample_prediction = best_model.predict(
    sample_features
)[0]


print("\n" + "=" * 70)
print("SAMPLE PREDICTION")
print("=" * 70)

print(
    "Actual:",
    sample_actual.upper()
)

print(
    "Predicted:",
    sample_prediction.upper()
)


# ==========================================================
# 17. CONFIDENCE
# ==========================================================

if hasattr(
    best_model,
    "predict_proba"
):

    sample_probability = (
        best_model.predict_proba(
            sample_features
        )[0]
    )

    sample_confidence = (
        max(sample_probability) * 100
    )

    print(
        f"Confidence: "
        f"{sample_confidence:.2f}%"
    )

elif hasattr(
    best_model,
    "decision_function"
):

    decision_score = (
        best_model.decision_function(
            sample_features
        )
    )

    # Convert SVM decision distance to
    # a simple 0-100 confidence indicator.

    confidence = (
        1 /
        (
            1 +
            np.exp(
                -abs(
                    float(
                        np.ravel(
                            decision_score
                        )[0]
                    )
                )
            )
        )
    ) * 100

    print(
        f"Confidence: "
        f"{confidence:.2f}%"
    )


print("\nTraining completed successfully!")

