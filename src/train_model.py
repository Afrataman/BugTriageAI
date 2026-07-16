import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# CSV veri setini oku
data = pd.read_csv("data/bug_reports.csv")


# X: Modelin inceleyeceği metinler
X = data["text"]

# y: Metinlerin doğru sınıfları
y = data["label"]


# Veriyi eğitim ve test bölümlerine ayır
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Eğitim ve test verisi sayılarını göster
print("Eğitim verisi sayısı:", len(X_train))
print("Test verisi sayısı:", len(X_test))


# Sınıfların dengeli dağılıp dağılmadığını göster
print("\nEğitim sınıf dağılımı:")
print(y_train.value_counts())

print("\nTest sınıf dağılımı:")
print(y_test.value_counts())


# Metinleri sayısal verilere çevirecek TF-IDF aracını oluştur
vectorizer = TfidfVectorizer()


# Eğitim metinlerindeki kelimeleri öğren ve sayısal verilere dönüştür
X_train_tfidf = vectorizer.fit_transform(X_train)


# Test metinlerini eğitim verisinden öğrenilen kelimelerle dönüştür
X_test_tfidf = vectorizer.transform(X_test)


# Oluşan sayısal tabloların boyutlarını göster
print("\nEğitim TF-IDF boyutu:", X_train_tfidf.shape)
print("Test TF-IDF boyutu:", X_test_tfidf.shape)


# Logistic Regression modelini oluştur
logistic_model = LogisticRegression(max_iter=1000)

# Modeli eğitim verisiyle eğit
logistic_model.fit(X_train_tfidf, y_train)

# Test verileri üzerinde tahmin yap
logistic_predictions = logistic_model.predict(X_test_tfidf)

# Doğruluk oranını hesapla
logistic_accuracy = accuracy_score(y_test, logistic_predictions)

print("\nLogistic Regression sonuçları:")
print("Gerçek sınıflar:", list(y_test))
print("Tahmin edilen sınıflar:", list(logistic_predictions))
print("Accuracy:", logistic_accuracy)

# Multinomial Naive Bayes modelini oluştur
naive_bayes_model = MultinomialNB()

# Modeli eğitim verisiyle eğit
naive_bayes_model.fit(X_train_tfidf, y_train)

# Test verileri üzerinde tahmin yap
naive_bayes_predictions = naive_bayes_model.predict(X_test_tfidf)

# Doğruluk oranını hesapla
naive_bayes_accuracy = accuracy_score(
    y_test,
    naive_bayes_predictions
)

print("\nMultinomial Naive Bayes sonuçları:")
print("Gerçek sınıflar:", list(y_test))
print("Tahmin edilen sınıflar:", list(naive_bayes_predictions))
print("Accuracy:", naive_bayes_accuracy)


print("\nLogistic Regression ayrıntılı sonuçları:")
print(classification_report(y_test, logistic_predictions))

print("\nNaive Bayes ayrıntılı sonuçları:")
print(classification_report(y_test, naive_bayes_predictions))


# Doğruluk oranlarına göre en iyi modeli seç
if logistic_accuracy >= naive_bayes_accuracy:
    best_model = logistic_model
    best_model_name = "Logistic Regression"
else:
    best_model = naive_bayes_model
    best_model_name = "Multinomial Naive Bayes"


# En iyi modeli models klasörüne kaydet
joblib.dump(best_model, "models/best_model.joblib")


# TF-IDF aracını models klasörüne kaydet
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")


print("\nKaydedilen en iyi model:", best_model_name)
print("Model ve TF-IDF dosyaları başarıyla kaydedildi.")

