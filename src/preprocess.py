import pandas as pd

file_path = "data/bug_reports.csv"

data = pd.read_csv(file_path)
data["text"] = data["text"].str.strip()
data["label"] = data["label"].str.strip()

print(data.head())
print("Satır sayısı:", data.shape[0])

print("Sütun sayısı:", data.shape[1])
print("\nEksik değer sayıları:")

print(data.isnull().sum())
print("\nTekrarlanan satır sayısı:")

print(data.duplicated().sum())

print("\nKategori dağılımı:")
print(data["label"].value_counts())