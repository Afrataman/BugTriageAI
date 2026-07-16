import joblib


# Daha önce kaydettiğimiz modeli yükle
model = joblib.load("models/best_model.joblib")


# Daha önce kaydettiğimiz TF-IDF aracını yükle
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")


# Tahmin etmek istediğimiz örnek metin
issue_text = "Login butonuna basınca uygulama kapanıyor."


# Metni TF-IDF ile sayısal veriye dönüştür
issue_tfidf = vectorizer.transform([issue_text])


# Metnin sınıfını tahmin et
prediction = model.predict(issue_tfidf)[0]


# Her sınıfa ait olasılıkları hesapla
probabilities = model.predict_proba(issue_tfidf)[0]


# En yüksek olasılığı güven oranı olarak al
confidence = max(probabilities) * 100


print("Girilen açıklama:", issue_text)
print("Tahmin edilen tür:", prediction)
print(f"Güven oranı: %{confidence:.2f}")