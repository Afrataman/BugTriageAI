import csv
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# DOSYA YOLLARI
# --------------------------------------------------

model_file = Path("models/best_model.joblib")
vectorizer_file = Path("models/tfidf_vectorizer.joblib")
results_file = Path("models/model_results.csv")
dataset_file = Path("data/bug_reports.csv")
feedback_file = Path("data/feedback.csv")


# --------------------------------------------------
# MODEL VE TF-IDF ARACINI YÜKLE
# --------------------------------------------------

model = joblib.load(model_file)
vectorizer = joblib.load(vectorizer_file)


# --------------------------------------------------
# SESSION STATE
# Son yapılan tahmini sayfa yenilendiğinde korur.
# --------------------------------------------------

if "last_text" not in st.session_state:
    st.session_state.last_text = ""

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = 0.0


# --------------------------------------------------
# SAYFA BAŞLIĞI
# --------------------------------------------------

st.title("BugTriage AI")

st.write(
    "Bir yazılım sorunu veya geliştirme talebi girin. "
    "Sistem kaydın türünü tahmin etsin."
)


# --------------------------------------------------
# VERİ SETİ ÖZETİ
# --------------------------------------------------

if dataset_file.exists():
    dataset = pd.read_csv(dataset_file)

    category_counts = dataset["label"].value_counts()

    with st.expander("Veri seti özeti"):
        st.write(f"Toplam kayıt sayısı: {len(dataset)}")
        st.bar_chart(category_counts)


# --------------------------------------------------
# MODEL KARŞILAŞTIRMA SONUÇLARI
# --------------------------------------------------

if results_file.exists():
    model_results = pd.read_csv(results_file)

    with st.expander("Model karşılaştırma sonuçları"):
        st.dataframe(
            model_results,
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "Not: Veri seti çok küçük olduğu için mevcut sonuçlar "
            "gerçek kullanım performansını kesin olarak göstermez."
        )


# --------------------------------------------------
# KULLANICI METİN GİRİŞİ
# --------------------------------------------------

issue_text = st.text_area(
    "Sorun açıklaması",
    placeholder=(
        "Örnek: Login butonuna basıldığında "
        "uygulama kapanıyor."
    )
)


# --------------------------------------------------
# TAHMİN BUTONU
# --------------------------------------------------

if st.button("Tahmin Et"):
    cleaned_text = issue_text.strip()

    if not cleaned_text:
        st.warning("Lütfen bir sorun açıklaması girin.")
        st.session_state.last_prediction = None

    else:
        # Kullanıcının metnini TF-IDF ile sayısal veriye çevir
        issue_tfidf = vectorizer.transform([cleaned_text])

        # Metnin sınıfını tahmin et
        prediction = model.predict(issue_tfidf)[0]

        # Her sınıfın olasılıklarını hesapla
        probabilities = model.predict_proba(issue_tfidf)[0]

        # En yüksek olasılığı güven oranı olarak al
        confidence = max(probabilities) * 100

        # Sonucu session state içinde sakla
        st.session_state.last_text = cleaned_text
        st.session_state.last_prediction = prediction
        st.session_state.last_confidence = confidence


# --------------------------------------------------
# TAHMİN SONUCUNU GÖSTER
# --------------------------------------------------

if st.session_state.last_prediction is not None:
    st.success(
        f"Tahmin edilen tür: "
        f"**{st.session_state.last_prediction}**"
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
            st.info(
                "Teşekkürler. Tahminin doğru olduğu bildirildi."
            )

        else:
            # Dosya yoksa veya boşsa başlık eklenmesi gerekir
            needs_header = (
                not feedback_file.exists()
                or feedback_file.stat().st_size == 0
            )

            # Geri bildirimi CSV dosyasına ekle
            with feedback_file.open(
                "a",
                newline="",
                encoding="utf-8-sig"
            ) as file:
                writer = csv.writer(file)

                if needs_header:
                    writer.writerow(
                        [
                            "text",
                            "predicted_label",
                            "correct_label"
                        ]
                    )

                writer.writerow(
                    [
                        st.session_state.last_text,
                        st.session_state.last_prediction,
                        correct_label
                    ]
                )

            st.success(
                "Yanlış tahmin geri bildirimi kaydedildi."
            )