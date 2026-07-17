import csv
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# SAYFA AYARLARI
# --------------------------------------------------

st.set_page_config(
    page_title="BugTriage AI",
    page_icon="🐞",
    layout="centered",
)


# --------------------------------------------------
# DOSYA YOLLARI
# --------------------------------------------------

# app.py dosyasının bulunduğu ana proje klasörü
project_root = Path(__file__).resolve().parent

model_file = project_root / "models" / "best_model.joblib"
vectorizer_file = project_root / "models" / "tfidf_vectorizer.joblib"
results_file = project_root / "models" / "model_results.csv"
metadata_file = project_root / "models" / "model_metadata.json"
confusion_matrix_file = project_root / "models" / "confusion_matrix.png"

dataset_file = project_root / "data" / "bug_reports.csv"
feedback_file = project_root / "data" / "feedback.csv"


# --------------------------------------------------
# GEREKLİ MODEL DOSYALARINI KONTROL ET
# --------------------------------------------------

required_model_files = [
    model_file,
    vectorizer_file,
]

missing_files = [
    file.name
    for file in required_model_files
    if not file.exists()
]

if missing_files:
    st.error(
        "Gerekli model dosyaları bulunamadı: "
        + ", ".join(missing_files)
    )

    st.info(
        "Önce terminalde şu komutu çalıştırın: "
        "`python src/train_model.py`"
    )

    st.stop()


# --------------------------------------------------
# MODEL VE TF-IDF ARACINI YÜKLE
# --------------------------------------------------

@st.cache_resource
def load_model_resources(
    model_path,
    vectorizer_path,
    model_update_time,
    vectorizer_update_time,
):
    """
    Eğitilmiş modeli ve TF-IDF aracını dosyadan yükler.

    Güncellenme zamanları parametre olarak verildiği için
    model yeniden eğitildiğinde önbellek de yenilenir.
    """

    loaded_model = joblib.load(model_path)
    loaded_vectorizer = joblib.load(vectorizer_path)

    return loaded_model, loaded_vectorizer


model, vectorizer = load_model_resources(
    model_file,
    vectorizer_file,
    model_file.stat().st_mtime,
    vectorizer_file.stat().st_mtime,
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

# Streamlit her buton tıklamasında kodu yeniden çalıştırır.
# Session state, son tahminin kaybolmasını engeller.

if "last_text" not in st.session_state:
    st.session_state.last_text = ""

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = 0.0

if "last_probabilities" not in st.session_state:
    st.session_state.last_probabilities = {}


# --------------------------------------------------
# SAYFA BAŞLIĞI
# --------------------------------------------------

st.title("🐞 BugTriage AI")

st.write(
    "Bir yazılım sorunu veya geliştirme talebi girin. "
    "Sistem kaydın türünü Bug, Feature veya "
    "Documentation olarak tahmin etsin."
)


# --------------------------------------------------
# VERİ SETİ ÖZETİ
# --------------------------------------------------

if dataset_file.exists():
    dataset = pd.read_csv(dataset_file)

    category_counts = (
        dataset["label"]
        .value_counts()
        .rename_axis("Sınıf")
        .to_frame("Kayıt Sayısı")
    )

    with st.expander("Veri seti özeti"):
        st.metric(
            "Toplam kayıt sayısı",
            len(dataset),
        )

        st.dataframe(
            category_counts,
            width="stretch",
        )

        st.bar_chart(category_counts)


# --------------------------------------------------
# MODEL KARŞILAŞTIRMA SONUÇLARI
# --------------------------------------------------

if results_file.exists():
    model_results = pd.read_csv(results_file)

    numeric_columns = model_results.select_dtypes(
        include="number"
    ).columns

    model_results[numeric_columns] = (
        model_results[numeric_columns].round(4)
    )

    with st.expander("Model karşılaştırma sonuçları"):
        st.dataframe(
            model_results,
            hide_index=True,
            width="stretch",
        )

        st.caption(
            "Modeller çapraz doğrulama sonuçlarına göre "
            "karşılaştırılmıştır. Veri seti hâlâ küçük olduğu "
            "için sonuçlar dikkatli yorumlanmalıdır."
        )


# --------------------------------------------------
# MODEL EĞİTİM ÖZETİ
# --------------------------------------------------

if metadata_file.exists():
    with metadata_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    with st.expander("Model eğitim özeti"):
        first_column, second_column = st.columns(2)

        first_column.metric(
            "En iyi model",
            metadata.get(
                "best_model",
                "Bilinmiyor",
            ),
        )

        second_column.metric(
            "Veri sayısı",
            metadata.get(
                "dataset_size",
                0,
            ),
        )

        classes = metadata.get("classes", [])

        st.write(
            "**Sınıflar:**",
            ", ".join(classes),
        )

        st.write(
            "**Model seçim ölçütü:**",
            metadata.get(
                "selection_metric",
                "Bilinmiyor",
            ),
        )

        st.write(
            "**Çapraz doğrulama:**",
            f"{metadata.get('cross_validation_folds', 0)} kat",
        )


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

if confusion_matrix_file.exists():
    with st.expander("Confusion matrix"):
        st.image(
            str(confusion_matrix_file),
            caption=(
                "En iyi modelin ayrılmış test verisi "
                "üzerindeki sonuçları"
            ),
        )

        st.caption(
            "Satırlar gerçek sınıfları, sütunlar ise "
            "modelin tahmin ettiği sınıfları gösterir."
        )


# --------------------------------------------------
# KULLANICI METİN GİRİŞİ
# --------------------------------------------------

issue_text = st.text_area(
    "Sorun açıklaması",
    placeholder=(
        "Örnek: Login butonuna basıldığında "
        "uygulama kapanıyor."
    ),
    height=140,
)


# --------------------------------------------------
# TAHMİN BUTONU
# --------------------------------------------------

if st.button(
    "Tahmin Et",
    type="primary",
):
    cleaned_text = issue_text.strip()

    if not cleaned_text:
        st.warning(
            "Lütfen bir sorun açıklaması girin."
        )

        st.session_state.last_prediction = None
        st.session_state.last_probabilities = {}

    else:
        # Metni TF-IDF ile sayısal veriye dönüştür
        issue_tfidf = vectorizer.transform(
            [cleaned_text]
        )

        # Sınıf tahmini yap
        prediction = model.predict(
            issue_tfidf
        )[0]

        # Her sınıfın olasılığını hesapla
        probabilities = model.predict_proba(
            issue_tfidf
        )[0]

        probability_by_class = {
            class_name: probability * 100
            for class_name, probability in zip(
                model.classes_,
                probabilities,
            )
        }

        # En yüksek olasılığı güven oranı olarak kullan
        confidence = max(probabilities) * 100

        # Sonucu session state içinde sakla
        st.session_state.last_text = cleaned_text
        st.session_state.last_prediction = prediction
        st.session_state.last_confidence = confidence
        st.session_state.last_probabilities = (
            probability_by_class
        )


# --------------------------------------------------
# TAHMİN SONUCUNU GÖSTER
# --------------------------------------------------

if st.session_state.last_prediction is not None:
    st.success(
        "Tahmin edilen tür: "
        f"**{st.session_state.last_prediction}**"
    )

    st.metric(
        "Güven oranı",
        f"%{st.session_state.last_confidence:.2f}",
    )

    # Güven oranı düşükse kullanıcıyı uyar
    if st.session_state.last_confidence < 60:
        st.warning(
            "Model bu tahminden yeterince emin değil. "
            "Sonuç kullanıcı tarafından kontrol edilmelidir."
        )

    # Bütün sınıfların olasılık tablosunu hazırla
    probability_table = pd.DataFrame(
        [
            {
                "Sınıf": class_name,
                "Olasılık (%)": round(
                    probability,
                    2,
                ),
            }
            for class_name, probability
            in st.session_state.last_probabilities.items()
        ]
    )

    probability_table = probability_table.sort_values(
        by="Olasılık (%)",
        ascending=False,
    )

    with st.expander("Tüm sınıf olasılıkları"):
        st.dataframe(
            probability_table,
            hide_index=True,
            width="stretch",
        )

        probability_chart = (
            probability_table
            .set_index("Sınıf")
        )

        st.bar_chart(probability_chart)

        st.caption(
            "Bunlar modelin ürettiği ham olasılık "
            "değerleridir. Veri seti küçük olduğu için "
            "kesin güven ölçüsü olarak değerlendirilmemelidir."
        )


    # --------------------------------------------------
    # KULLANICI GERİ BİLDİRİMİ
    # --------------------------------------------------

    st.subheader("Tahmin yanlış mı?")

    predicted_label = (
        st.session_state.last_prediction
    )

    # Tahmin edilen sınıf dışındaki seçenekleri göster
    alternative_labels = [
        class_name
        for class_name in model.classes_
        if class_name != predicted_label
    ]

    correct_label = st.selectbox(
        "Doğru sınıfı seçin",
        [
            "Tahmin doğru",
            *alternative_labels,
        ],
    )

    if st.button("Geri Bildirimi Kaydet"):
        if correct_label == "Tahmin doğru":
            st.info(
                "Teşekkürler. Tahminin doğru olduğu "
                "bildirildi."
            )

        else:
            # Dosya yoksa veya boşsa başlık yazılacak
            needs_header = (
                not feedback_file.exists()
                or feedback_file.stat().st_size == 0
            )

            feedback_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Geri bildirimi dosyanın sonuna ekle
            with feedback_file.open(
                "a",
                newline="",
                encoding="utf-8-sig",
            ) as file:
                writer = csv.writer(file)

                if needs_header:
                    writer.writerow(
                        [
                            "text",
                            "predicted_label",
                            "correct_label",
                        ]
                    )

                writer.writerow(
                    [
                        st.session_state.last_text,
                        predicted_label,
                        correct_label,
                    ]
                )

            st.success(
                "Yanlış tahmin geri bildirimi kaydedildi."
            )