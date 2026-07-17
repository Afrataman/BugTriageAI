import csv
from pathlib import Path

import joblib
import streamlit as st


# Kaydedilmiş model ve TF-IDF aracını yükle
model = joblib.load("models/best_model.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

# Geri bildirimlerin kaydedileceği dosya
feedback_file = Path("data/feedback.csv")


# Son tahmini sayfa yenilense bile hatırlamak için
if "last_text" not in st.session_state:
    st.session_state.last_text = ""

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = 0.0


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
    cleaned_text = issue_text.strip()

    if not cleaned_text:
        st.warning("Lütfen bir sorun açıklaması girin.")
        st.session_state.last_prediction = None

    else:
        # Metni sayısal TF-IDF verisine dönüştür
        issue_tfidf = vectorizer.transform([cleaned_text])

        # Sınıf tahmini yap
        prediction = model.predict(issue_tfidf)[0]

        # Güven oranını hesapla
        probabilities = model.predict_proba(issue_tfidf)[0]
        confidence = max(probabilities) * 100

        # Sonucu hafızada sakla
        st.session_state.last_text = cleaned_text
        st.session_state.last_prediction = prediction
        st.session_state.last_confidence = confidence


# Daha önce tahmin yapıldıysa sonucu göster
if st.session_state.last_prediction is not None:
    st.success(
        f"Tahmin edilen tür: **{st.session_state.last_prediction}**"
    )

    st.metric(
        "Güven oranı",
        f"%{st.session_state.last_confidence:.2f}"
    )

    st.subheader("Tahmin yanlış mı?")

    correct_label = st.selectbox(
        "Doğru sınıfı seçin",
        [
            "Tahmin doğru",
            "Bug",
            "Feature",
            "Documentation"
        ]
    )

    if st.button("Geri Bildirimi Kaydet"):

        if correct_label == "Tahmin doğru":
            st.info("Teşekkürler. Tahminin doğru olduğu bildirildi.")

        else:
            # Dosya boşsa başlık satırı eklenecek
            needs_header = (
                not feedback_file.exists()
                or feedback_file.stat().st_size == 0
            )

            # Yeni geri bildirimi CSV dosyasının sonuna ekle
            with feedback_file.open(
                "a",
                newline="",
                encoding="utf-8-sig"
            ) as file:
                writer = csv.writer(file)

                if needs_header:
                    writer.writerow(
                        ["text", "predicted_label", "correct_label"]
                    )

                writer.writerow(
                    [
                        st.session_state.last_text,
                        st.session_state.last_prediction,
                        correct_label
                    ]
                )

            st.success("Yanlış tahmin geri bildirimi kaydedildi.")