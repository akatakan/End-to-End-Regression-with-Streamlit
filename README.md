# Ev Kirası Tahmini

Türkiye'deki kiralık ev ilanlarından toplanan veriyle kira fiyatı tahmin eden
uçtan uca bir proje: veri kazıma → temizlik → CatBoost regresyon → Streamlit arayüzü.

## Kurulum ve çalıştırma

```bash
pip install -r requirements.txt
python train.py --data emlak.csv     # model_pipeline.pkl + metrics.json + locations.json üretir
streamlit run app.py
```

Python 3.11+ gerekir.

## Model performansı

8.747 ilandan 8.357'si temizlikten geçiyor, %20'si test için ayrılıyor.
Aykırı değer kırpması **sadece eğitim setine** uygulanıyor; test seti dokunulmadan
bırakıldığı için aşağıdaki sayılar gerçekte görülecek performansı temsil ediyor.

| | n | MAPE | MedAPE | R² | ±%20 bandında |
|---|---|---|---|---|---|
| Medyan baseline | 1.672 | %47,7 | %33,3 | −0,10 | %32,8 |
| CatBoost — tüm test | 1.672 | %27,1 | %17,8 | 0,12 | %55,4 |
| CatBoost — eğitim aralığı içi (4.463–94.101 TL) | 1.514 | %19,8 | %15,4 | **0,685** | %61,0 |

Tüm test satırlarındaki düşük R², dağılımın ucundaki (94.101 TL üstü) ilanlardan
kaynaklanıyor; model o aralıkta eğitilmediği için arayüz böyle bir tahmin
ürettiğinde uyarı gösteriyor.

En önemli özellikler: `Town` (25,2) · `City` (12,2) · `House Size` (10,8) ·
`Bathroom Count` (9,9) · `Furniture` (5,3). Tam liste `metrics.json` içinde.

## Dosyalar

| Dosya | İçerik |
|---|---|
| `features.py` | Özellik üretimi + `CatBoostDF`. Pipeline pickle'ı buraya referans verdiği için ayrı modülde. |
| `train.py` | Temizlik, split, eğitim, değerlendirme, artefakt üretimi. |
| `compare.py` | Notebook yöntemi ile yeni yöntemin aynı split üzerinde karşılaştırması. |
| `app.py` | Streamlit arayüzü. Seçenekler `metrics.json` / `locations.json`'dan geliyor. |
| `scrape_common.py` | Kazıma altyapısı: geri çekilmeli HTTP, robots kontrolü, Türkçe sayı ayrıştırma, checkpoint. |
| `link_scrap.py`, `info_scrapy.py` | Veri kazıma (`pip install -r requirements-scraper.txt`). |
| `test_scraper.py` | Kazıma mantığının testleri — ağa çıkmaz (`python test_scraper.py`). |
| `emlak.csv` | 8.747 ilanlık ham veri seti. |
| `rent-regression.ipynb` | İlk keşif defteri (üretim yolu artık `train.py`). |

## Veri temizliğinde ne yapılıyor

Ham veride üç tür kirlilik var ve üçü de kaynağında düzeltiliyor:

- **Binlik ayırıcı hatası:** kazıyıcı `"1.100 m²"` metnini olduğu gibi bıraktığı
  için 1.000 m² üstü her ev `1.1` gibi okunuyordu. 20 m²'nin altındaki değerler
  1000 ile çarpılıyor (`features.py` bunu tahmin sırasında da uyguluyor).
- **Para birimi:** kazıyıcı fiyattaki sembolü düşürüyor; K.K.T.C. ilanları
  GBP/EUR cinsinden ve medyanları 925 TL görünüyor. Bu satırlar ayıklanıyor.
- **Fiyat uçları:** 1–999 TL arası tuzak ilanlar ve 1.000.000 TL üstü
  yıllık/satılık karışımları atılıyor. Kalan aykırı değerler **log ekseninde**
  IQR ile kırpılıyor — ham IQR bu log-normal dağılımda alt sınırı negatife
  itip gerçek çöpü kaçırıyor, üstten ise meşru pahalı ilanları biçiyordu.

318 tam duplike satır siliniyor; bunlar daha önce hem eğitim hem test setine
düşüp skoru şişiriyordu.

## Veri kazıma

```bash
pip install -r requirements-scraper.txt
python link_scrap.py --pages 1892 --resume     # ilan linkleri  -> links.json
python info_scrapy.py --links links.json       # ilan detayları -> listings.jsonl + emlak.csv
python info_scrapy.py --csv-only               # yeniden kazımadan CSV'yi üret
python test_scraper.py                         # 28 test, ağa çıkmaz
```

Her iki script de **kaldığı yerden devam eder** (`--resume` / işlenmiş URL kümesi),
her kaydı anında diske yazar ve robots.txt'e sorar. Ham HTML alanları
`listings.jsonl` içinde saklandığı için alan çıkarımı değişirse siteyi tekrar
kazımaya gerek kalmaz — `--csv-only` yeter.

> **2026-09 durumu:** hepsiemlak Cloudflare arkasına alınmış ve düz HTTP
> istemcilerine `HTTP 403` dönüyor. robots.txt taranan yolları yasaklamıyor
> (`Allow: /`, `Crawl-delay: 2`) ama erişim tarayıcı doğrulamasına tabi. Yani
> scriptler mantık olarak doğru çalışıyor, siteye erişim ayrı bir mesele —
> yeni veri toplamadan önce siteyle iletişime geçmek gerekir. `test_scraper.py`
> içindeki HTML fixture'ları eski koddan türetildi ve **canlı sayfaya karşı
> doğrulanamadı**; seçiciler değiştiyse önce orası güncellenmeli.

## Bilinen sınırlar

- Veri tek seferlik bir kazımadan geliyor; kira fiyatları hızlı değiştiği için
  model zamanla eskiyor. Tarih bilgisi `metrics.json > trained_at` içinde.
- Mahalle bilgisi en güçlü sinyal ama 547 mahalle veride yalnızca bir kez
  geçiyor; arayüz sadece eğitim verisinde bulunan konumları listeliyor.
- Model 4.463–94.101 TL aralığında eğitildi; dışına çıkan tahminlere güvenilmez.
- Kazıma scriptleri hepsiemlak'ın sayfa yapısına bağlı ve tek seferlik kullanım
  için yazıldı; sitenin kullanım koşullarını kontrol etmeden çalıştırmayın.

## Lisans

MIT — bkz. `LICENSE`.
