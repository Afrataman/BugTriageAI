from pathlib import Path

import pandas as pd


# Projenin ana klasörünü bul
project_root = Path(__file__).resolve().parent.parent

# Veri setinin konumu
dataset_file = project_root / "data" / "bug_reports.csv"


# Veri setine eklenecek yeni ve etiketli örnekler
new_records = [
    # --------------------------------------------------
    # BUG ÖRNEKLERİ
    # --------------------------------------------------
    {
        "text": "Filtre değiştirildiğinde arama sonuçları yenilenmiyor.",
        "label": "Bug",
    },
    {
        "text": "Şifre sıfırlama e-postasındaki bağlantı 404 hatası veriyor.",
        "label": "Bug",
    },
    {
        "text": "Büyük bir dosya yüklenirken tarayıcı donuyor.",
        "label": "Bug",
    },
    {
        "text": "Dışa aktarılan raporda Türkçe karakterler bozuk görünüyor.",
        "label": "Bug",
    },
    {
        "text": "Bildirim sayacı bazı kullanıcılarda negatif değer gösteriyor.",
        "label": "Bug",
    },
    {
        "text": "Sayfa yenilendiğinde sepetteki ürünler kayboluyor.",
        "label": "Bug",
    },
    {
        "text": "Seçilen tarih kaydedilirken bir önceki gün olarak görünüyor.",
        "label": "Bug",
    },
    {
        "text": "Mobil klavye form gönderme butonunun üzerini kapatıyor.",
        "label": "Bug",
    },
    {
        "text": "Aynı sipariş için iki farklı fatura numarası oluşturuluyor.",
        "label": "Bug",
    },
    {
        "text": "Yetkisi olmayan kullanıcı yönetici butonunu görebiliyor.",
        "label": "Bug",
    },
    {
        "text": "Silinen ürün arama sonuçlarında görünmeye devam ediyor.",
        "label": "Bug",
    },
    {
        "text": "İndirim kodu ödeme sırasında iki kez uygulanıyor.",
        "label": "Bug",
    },
    {
        "text": "Google ile giriş yapıldıktan sonra yönlendirme döngüye giriyor.",
        "label": "Bug",
    },
    {
        "text": "PDF raporunun son sayfası oluşturulmuyor.",
        "label": "Bug",
    },
    {
        "text": "Otomatik kaydetme daha yeni olan değişikliklerin üzerine yazıyor.",
        "label": "Bug",
    },
    {
        "text": "Oturum süresi dolduğunda kullanıcı giriş ekranına yönlendirilmiyor.",
        "label": "Bug",
    },
    {
        "text": "Arama işlemi büyük ve küçük harflere göre farklı sonuç veriyor.",
        "label": "Bug",
    },
    {
        "text": "Saat dilimi değiştirildiğinde rapor grafiği yanlış verileri gösteriyor.",
        "label": "Bug",
    },
    {
        "text": "Geri tuşuna basıldığında parola alanındaki değer görünür oluyor.",
        "label": "Bug",
    },
    {
        "text": "İsteğe bağlı alan boş gönderildiğinde API 500 hatası döndürüyor.",
        "label": "Bug",
    },

    # --------------------------------------------------
    # FEATURE ÖRNEKLERİ
    # --------------------------------------------------
    {
        "text": "Kullanıcıların iki aşamalı doğrulamayı etkinleştirebilmesi sağlanmalı.",
        "label": "Feature",
    },
    {
        "text": "Arama geçmişini temizlemek için bir buton eklenmeli.",
        "label": "Feature",
    },
    {
        "text": "Sipariş listesine Excel olarak dışa aktarma özelliği eklenmeli.",
        "label": "Feature",
    },
    {
        "text": "Kullanıcılar profil fotoğraflarını kırpabilmeli.",
        "label": "Feature",
    },
    {
        "text": "Uygulamaya açık ve koyu tema arasında otomatik geçiş eklenmeli.",
        "label": "Feature",
    },
    {
        "text": "Bildirimler okunmadı olarak işaretlenebilmelidir.",
        "label": "Feature",
    },
    {
        "text": "Raporlar belirli tarihlerde otomatik olarak e-posta ile gönderilmeli.",
        "label": "Feature",
    },
    {
        "text": "Kullanıcıların birden fazla teslimat adresi kaydedebilmesi sağlanmalı.",
        "label": "Feature",
    },
    {
        "text": "Ürün karşılaştırma ekranı eklenmeli.",
        "label": "Feature",
    },
    {
        "text": "Yönetici paneline toplu kullanıcı silme özelliği eklenmeli.",
        "label": "Feature",
    },
    {
        "text": "Formlara taslak olarak kaydetme seçeneği eklenmeli.",
        "label": "Feature",
    },
    {
        "text": "Kullanıcılar uygulama dilini profil ayarlarından değiştirebilmeli.",
        "label": "Feature",
    },
    {
        "text": "Arama sonuçları puana göre sıralanabilmeli.",
        "label": "Feature",
    },
    {
        "text": "Projelerde ekip üyelerini etiketleme özelliği olmalı.",
        "label": "Feature",
    },
    {
        "text": "Sisteme takvim görünümü eklenmeli.",
        "label": "Feature",
    },
    {
        "text": "Kullanıcıların kendi bildirim seslerini seçebilmesi sağlanmalı.",
        "label": "Feature",
    },
    {
        "text": "Yüklenen dosyalar için ön izleme ekranı eklenmeli.",
        "label": "Feature",
    },
    {
        "text": "Kullanıcı etkinliklerini gösteren zaman çizelgesi oluşturulmalı.",
        "label": "Feature",
    },
    {
        "text": "Raporlara özel tarih aralığı filtresi eklenmeli.",
        "label": "Feature",
    },
    {
        "text": "Kullanıcıların hesap verilerini indirebilmesi sağlanmalı.",
        "label": "Feature",
    },

    # --------------------------------------------------
    # DOCUMENTATION ÖRNEKLERİ
    # --------------------------------------------------
    {
        "text": "JWT yapılandırma adımları geliştirici dokümanında açıklanmalı.",
        "label": "Documentation",
    },
    {
        "text": "Docker ile çalıştırma komutları README dosyasına eklenmeli.",
        "label": "Documentation",
    },
    {
        "text": "API kimlik doğrulama örnekleri dokümantasyonda gösterilmeli.",
        "label": "Documentation",
    },
    {
        "text": "Veri tabanı yedekleme işlemi kullanıcı kılavuzunda anlatılmalı.",
        "label": "Documentation",
    },
    {
        "text": "Geliştirme ortamının kurulumu adım adım belgelenmeli.",
        "label": "Documentation",
    },
    {
        "text": "Her ortam değişkeninin görevi README dosyasında açıklanmalı.",
        "label": "Documentation",
    },
    {
        "text": "API cevap kodları için açıklayıcı bir tablo hazırlanmalı.",
        "label": "Documentation",
    },
    {
        "text": "Yeni geliştiriciler için projeye katkı rehberi yazılmalı.",
        "label": "Documentation",
    },
    {
        "text": "Şifre politikaları güvenlik dokümanına eklenmeli.",
        "label": "Documentation",
    },
    {
        "text": "Model eğitim komutları proje kılavuzunda açıklanmalı.",
        "label": "Documentation",
    },
    {
        "text": "Streamlit uygulamasının başlatılma adımları belgelenmeli.",
        "label": "Documentation",
    },
    {
        "text": "CSV veri setindeki sütunların anlamı dokümana eklenmeli.",
        "label": "Documentation",
    },
    {
        "text": "Geri bildirim dosyasının nasıl kullanıldığı açıklanmalı.",
        "label": "Documentation",
    },
    {
        "text": "Üretim ortamına yayınlama adımları README dosyasına yazılmalı.",
        "label": "Documentation",
    },
    {
        "text": "Sık kullanılan Git komutları katkı rehberine eklenmeli.",
        "label": "Documentation",
    },
    {
        "text": "Model değerlendirme metriklerinin anlamı açıklanmalı.",
        "label": "Documentation",
    },
    {
        "text": "Confusion matrix görselinin nasıl yorumlanacağı belgelenmeli.",
        "label": "Documentation",
    },
    {
        "text": "Desteklenen Python sürümleri kurulum sayfasında belirtilmeli.",
        "label": "Documentation",
    },
    {
        "text": "Bilinen sorunlar için ayrı bir dokümantasyon bölümü hazırlanmalı.",
        "label": "Documentation",
    },
    {
        "text": "Proje lisansı ve kullanım koşulları README dosyasında açıklanmalı.",
        "label": "Documentation",
    },
]


# Mevcut veri setini oku
existing_data = pd.read_csv(dataset_file)

# Yeni örnekleri tabloya dönüştür
new_data = pd.DataFrame(new_records)

# Eski ve yeni verileri birleştir
combined_data = pd.concat(
    [existing_data, new_data],
    ignore_index=True,
)

# Aynı metin ve etiket tekrar eklenmişse kaldır
combined_data = combined_data.drop_duplicates(
    subset=["text", "label"]
)

# Temizlenmiş veri setini tekrar kaydet
combined_data.to_csv(
    dataset_file,
    index=False,
    encoding="utf-8-sig",
)


print("Önceki kayıt sayısı:", len(existing_data))
print("Yeni toplam kayıt sayısı:", len(combined_data))

print("\nKategori dağılımı:")
print(combined_data["label"].value_counts())