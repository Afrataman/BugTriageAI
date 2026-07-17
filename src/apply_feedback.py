from pathlib import Path
import shutil

import pandas as pd


# --------------------------------------------------
# DOSYA YOLLARI
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

dataset_file = project_root / "data" / "bug_reports.csv"
feedback_file = project_root / "data" / "feedback.csv"
backup_file = project_root / "data" / "bug_reports_backup.csv"


# --------------------------------------------------
# DOSYALARI KONTROL ET
# --------------------------------------------------

if not dataset_file.exists():
    raise FileNotFoundError(
        "bug_reports.csv dosyası bulunamadı."
    )

if not feedback_file.exists():
    raise FileNotFoundError(
        "feedback.csv dosyası bulunamadı."
    )


# --------------------------------------------------
# VERİLERİ OKU
# --------------------------------------------------

dataset = pd.read_csv(dataset_file)
feedback = pd.read_csv(feedback_file)


# feedback.csv yalnızca başlık içeriyorsa kayıt yoktur
if feedback.empty:
    print("Uygulanacak yeni geri bildirim bulunamadı.")
    raise SystemExit


# --------------------------------------------------
# GEÇERLİ GERİ BİLDİRİMLERİ SEÇ
# --------------------------------------------------

required_columns = {
    "text",
    "predicted_label",
    "correct_label",
}

if not required_columns.issubset(feedback.columns):
    raise ValueError(
        "feedback.csv gerekli sütunları içermiyor."
    )


valid_labels = {
    "Bug",
    "Feature",
    "Documentation",
}


feedback = feedback.dropna(
    subset=["text", "correct_label"]
)

feedback["text"] = feedback["text"].str.strip()
feedback["correct_label"] = (
    feedback["correct_label"].str.strip()
)

feedback = feedback[
    feedback["correct_label"].isin(valid_labels)
]


if feedback.empty:
    print("Geçerli geri bildirim bulunamadı.")
    raise SystemExit


# --------------------------------------------------
# YEDEK AL
# --------------------------------------------------

shutil.copy2(
    dataset_file,
    backup_file,
)

print(
    "Veri setinin yedeği oluşturuldu:",
    backup_file.name,
)


# --------------------------------------------------
# GERİ BİLDİRİMLERİ EĞİTİM FORMATINA ÇEVİR
# --------------------------------------------------

corrected_records = feedback[
    ["text", "correct_label"]
].rename(
    columns={
        "correct_label": "label"
    }
)


# Mevcut veri ile düzeltilmiş kayıtları birleştir
updated_dataset = pd.concat(
    [
        dataset,
        corrected_records,
    ],
    ignore_index=True,
)


# Metinleri temizle
updated_dataset["text"] = (
    updated_dataset["text"]
    .astype(str)
    .str.strip()
)

updated_dataset["label"] = (
    updated_dataset["label"]
    .astype(str)
    .str.strip()
)


# Aynı metin birden fazla kez varsa son düzeltmeyi kullan
updated_dataset = updated_dataset.drop_duplicates(
    subset=["text"],
    keep="last",
)


# --------------------------------------------------
# GÜNCEL VERİ SETİNİ KAYDET
# --------------------------------------------------

updated_dataset.to_csv(
    dataset_file,
    index=False,
    encoding="utf-8-sig",
)


# İşlenen geri bildirimleri temizle
empty_feedback = pd.DataFrame(
    columns=[
        "text",
        "predicted_label",
        "correct_label",
    ]
)

empty_feedback.to_csv(
    feedback_file,
    index=False,
    encoding="utf-8-sig",
)


print(
    "İşlenen geri bildirim sayısı:",
    len(corrected_records),
)

print(
    "Yeni toplam eğitim kaydı:",
    len(updated_dataset),
)

print("\nYeni kategori dağılımı:")
print(updated_dataset["label"].value_counts())

print(
    "\nGeri bildirim dosyası temizlendi."
)