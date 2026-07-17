import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# Projenin ana klasörünü bul
project_root = Path(__file__).resolve().parent.parent

data_file = project_root / "data" / "bug_reports.csv"
models_folder = project_root / "models"

models_folder.mkdir(exist_ok=True)


# Veri setini oku ve temel temizliği yap
data = pd.read_csv(data_file)

data = data.dropna(subset=["text", "label"])
data["text"] = data["text"].str.strip()
data["label"] = data["label"].str.strip()
data = data.drop_duplicates()


X = data["text"]
y = data["label"]


# Son kontrol için ayrı test verisi oluştur
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# Modeller
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),
    "Multinomial Naive Bayes": MultinomialNB(),
}


# 5 katlı ve sınıf dengeli çapraz doğrulama
cross_validation = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


scoring = [
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
]


results = []
trained_models = {}
predictions = {}


for model_name, classifier in models.items():

    # Çapraz doğrulamada TF-IDF ve model birlikte çalışır
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            ("classifier", classifier),
        ]
    )

    cv_scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cross_validation,
        scoring=scoring,
    )

    # Ayrı test verisi için TF-IDF oluştur
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    classifier.fit(X_train_tfidf, y_train)

    model_predictions = classifier.predict(X_test_tfidf)

    accuracy = accuracy_score(
        y_test,
        model_predictions,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        model_predictions,
        average="weighted",
        zero_division=0,
    )

    trained_models[model_name] = classifier
    predictions[model_name] = model_predictions

    results.append(
        {
            "Model": model_name,
            "Holdout Accuracy": accuracy,
            "Holdout Precision": precision,
            "Holdout Recall": recall,
            "Holdout F1": f1,
            "CV Accuracy Mean": cv_scores["test_accuracy"].mean(),
            "CV Accuracy Std": cv_scores["test_accuracy"].std(),
            "CV F1 Mean": cv_scores["test_f1_weighted"].mean(),
            "CV F1 Std": cv_scores["test_f1_weighted"].std(),
        }
    )


results_table = pd.DataFrame(results)
results_table = results_table.round(4)

results_table.to_csv(
    models_folder / "model_results.csv",
    index=False,
)


# En iyi modeli çapraz doğrulama F1 değerine göre seç
best_result = max(
    results,
    key=lambda result: result["CV F1 Mean"],
)

best_model_name = best_result["Model"]


# En iyi modeli tüm veriyle yeniden eğit
final_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True,
)

X_all_tfidf = final_vectorizer.fit_transform(X)

if best_model_name == "Logistic Regression":
    final_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )
else:
    final_model = MultinomialNB()

final_model.fit(X_all_tfidf, y)


joblib.dump(
    final_model,
    models_folder / "best_model.joblib",
)

joblib.dump(
    final_vectorizer,
    models_folder / "tfidf_vectorizer.joblib",
)


# En iyi modelin test tahminleriyle confusion matrix oluştur
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions[best_model_name],
    labels=sorted(y.unique()),
)

plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()

plt.savefig(
    models_folder / "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# Model hakkındaki özet bilgileri kaydet
metadata = {
    "best_model": best_model_name,
    "dataset_size": len(data),
    "classes": sorted(y.unique().tolist()),
    "selection_metric": "Cross-validation weighted F1",
    "cross_validation_folds": 5,
}

with open(
    models_folder / "model_metadata.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metadata,
        file,
        ensure_ascii=False,
        indent=4,
    )


print("\nModel karşılaştırma sonuçları:")
print(results_table.to_string(index=False))

print("\nSeçilen en iyi model:", best_model_name)
print("Toplam veri sayısı:", len(data))
print("Model dosyaları başarıyla güncellendi.")