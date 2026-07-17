import json
from pathlib import Path

import joblib
import pandas as pd
import pytest


project_root = Path(__file__).resolve().parent.parent

dataset_file = project_root / "data" / "bug_reports.csv"
feedback_file = project_root / "data" / "feedback.csv"

model_file = project_root / "models" / "best_model.joblib"
vectorizer_file = (
    project_root / "models" / "tfidf_vectorizer.joblib"
)
metadata_file = (
    project_root / "models" / "model_metadata.json"
)


valid_labels = {
    "Bug",
    "Feature",
    "Documentation",
}


def test_dataset_file_exists():
    assert dataset_file.exists()


def test_dataset_columns_are_correct():
    dataset = pd.read_csv(dataset_file)

    assert list(dataset.columns) == [
        "text",
        "label",
    ]


def test_dataset_has_no_missing_values():
    dataset = pd.read_csv(dataset_file)

    assert dataset["text"].isnull().sum() == 0
    assert dataset["label"].isnull().sum() == 0


def test_dataset_labels_are_valid():
    dataset = pd.read_csv(dataset_file)

    dataset_labels = set(dataset["label"].unique())

    assert dataset_labels == valid_labels


def test_dataset_has_enough_examples():
    dataset = pd.read_csv(dataset_file)

    label_counts = dataset["label"].value_counts()

    for label in valid_labels:
        assert label_counts[label] >= 25


def test_feedback_file_columns_are_correct():
    feedback = pd.read_csv(feedback_file)

    assert list(feedback.columns) == [
        "text",
        "predicted_label",
        "correct_label",
    ]


def test_model_files_exist():
    assert model_file.exists()
    assert vectorizer_file.exists()


def test_model_can_make_prediction():
    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)

    sample_text = (
        "Giriş butonuna basıldığında uygulama kapanıyor."
    )

    sample_tfidf = vectorizer.transform(
        [sample_text]
    )

    prediction = model.predict(
        sample_tfidf
    )[0]

    assert prediction in valid_labels


def test_prediction_probabilities_are_valid():
    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)

    sample_text = (
        "Uygulamaya yeni bir filtre özelliği eklenmeli."
    )

    sample_tfidf = vectorizer.transform(
        [sample_text]
    )

    probabilities = model.predict_proba(
        sample_tfidf
    )[0]

    assert len(probabilities) == 3

    assert sum(probabilities) == pytest.approx(
        1.0,
        abs=0.0001,
    )


def test_model_metadata_is_valid():
    assert metadata_file.exists()

    with metadata_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    assert metadata["best_model"] in {
        "Logistic Regression",
        "Multinomial Naive Bayes",
    }

    assert metadata["dataset_size"] >= 90

    assert set(metadata["classes"]) == valid_labels
    