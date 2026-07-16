import joblib
import streamlit as st


# Kaydettiğimiz modeli yükle
model = joblib.load("models/best_model.joblib")

# Kaydettiğimiz TF-IDF aracını yükle
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")


st.title("BugTriage AI")

st.write(
    "Bir yazılım sorunu veya geliştirme talebi girin. "
    "Sistem kaydın türünü tahmin etsin."
)


issue_text = st.text_area(
    "Sorun açıklaması",
    placeholder="Örnek: Login butonuna basıldığında uygulama kapanıyor."
)


if st.button("Tahmin Et"):

    # Kullanıcı boş metin girdiyse uyarı göster
    if not issue_text.strip():
        st.warning("Lütfen bir sorun açıklaması girin.")

    else:
        # Kullanıcının metnini sayısal TF-IDF verisine dönüştür
        issue_tfidf = vectorizer.transform([issue_text])

        # Tür tahmini yap
        prediction = model.predict(issue_tfidf)[0]

        # Sınıfların olasılıklarını hesapla
        probabilities = model.predict_proba(issue_tfidf)[0]

        # En yüksek olasılığı güven oranı olarak al
        confidence = max(probabilities) * 100

        # Sonuçları ekranda göster
        st.success(f"Tahmin edilen tür: **{prediction}**")
        st.metric("Güven oranı", f"%{confidence:.2f}")