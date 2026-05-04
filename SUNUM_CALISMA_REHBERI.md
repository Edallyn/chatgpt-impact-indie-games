# SUNUM ÇALIŞMA REHBERİ
## ChatGPT'nin Steam Indie Oyunlarına Etkisi — ML Modeli Sunumu

> **Bu döküman sıfırdan hazırlanmıştır. Projeyi hiç bilmiyormuşsun gibi her şey açıklanmıştır.**

---

## BÖLÜM 0 — ARAŞTIRMA SORUSU (Sunumun İlk 1 Dakikası)

**Ana soru:**
> "ChatGPT'nin piyasaya sürülmesi (Kasım 2022) Steam'deki indie oyunların review (değerlendirme) skorlarını sistematik olarak etkiledi mi?"

**Bunu nasıl söylersin:**
"Bu çalışmada, ChatGPT lansmanının Steam indie oyun review skorları üzerinde ölçülebilir bir etki bırakıp bırakmadığını araştırdık. İki farklı yaklaşım kullandık: (1) Regresyon modeli ile oyun özelliklerinden skor tahmini, (2) Zaman serisi analizi ile yapısal kırılma testi."

---

## BÖLÜM 1 — VERİ SETİ HAKKINDA (Kısa hatırlatma)

> Not: Hoca EDA'yı tekrar anlatmamızı istemedi ama sorarsa hazır ol.

| Özellik | Değer |
|---------|-------|
| Kaynak | Steam API (indie oyunlar) |
| Toplam oyun | 4.363 oyun |
| Filtre | ≥10 review alan oyunlar |
| Eğitim/Test | %80 eğitim (3.490) / %20 test (873) |
| Hedef değişken | `review_wilson_lower` (0–1 arası skor) |

**Hedef değişken neden Wilson alt sınırı?**
Normal review oranı (örn. %90 pozitif) az yorumlu oyunlara avantaj sağlar. Wilson alt sınırı, istatistiksel güven aralığını dikkate alır — az yorum olan oyunlar için daha düşük, çok yorum olan oyunlar için daha gerçekçi skor üretir.

---

## BÖLÜM 2 — ÖZELLİKLER (14 Feature)

| Feature Adı | Türkçe Açıklama | Tür |
|-------------|-----------------|-----|
| `dev_game_count` | Geliştiricinin Steam'deki toplam oyun sayısı | Sayısal |
| `solo_dev_proxy` | Tek kişilik geliştirici mi? (1=Evet) | İkili |
| `dev_equals_publisher` | Geliştirici = Yayıncı mı? (1=Evet) | İkili |
| `achievement_count` | Steam başarım sayısı | Sayısal |
| `has_demo` | Demo var mı? (1=Evet) | İkili |
| `dlc_count` | DLC (ek içerik) sayısı | Sayısal |
| `genre_count` | Tür sayısı | Sayısal |
| `platform_count` | Desteklenen platform sayısı (Win/Mac/Linux) | Sayısal |
| `language_count` | Desteklenen dil sayısı | Sayısal |
| `price_usd` | Fiyat (USD) | Sayısal |
| `is_early_access` | Early Access (erken erişim) mi? (1=Evet) | İkili |
| `post_chatgpt` | ChatGPT sonrası mı? (Kasım 2022+: 1) | İkili |
| `release_year` | Çıkış yılı | Sayısal |
| `has_workshop` | Steam Workshop var mı? (1=Evet) | İkili |

**Önemli not:** `post_chatgpt` doğrudan araştırma sorusunu test eden değişkendir. 

---

## BÖLÜM 3 — REGRESYON MODELLERİ

### 3.1 Hangi Modelleri Denedik ve Neden?

#### DOĞRUSAL MODELLER (Baseline / Başlangıç Noktası)

**Linear Regression (Doğrusal Regresyon)**
- En basit model. Feature'lar ile hedef arasındaki ilişkinin düz bir çizgi olduğunu varsayar.
- Neden denedik? Her projede baseline olarak kullanılır. Daha karmaşık modellerin bunu yenmesi gerekir.
- Sonuç: CV R² = 0.1114 ± 0.0130

**Ridge Regression**
- Doğrusal regresyon + L2 cezası. Büyük katsayıları küçültür, overfitting'i önler.
- Sonuç: CV R² = 0.1114 ± 0.0130 (Linear ile neredeyse aynı → multicollinearity yok)

**Lasso Regression**
- Doğrusal regresyon + L1 cezası. Gereksiz feature katsayılarını sıfıra indirir (otomatik feature seçimi).
- Sonuç: CV R² = 0.0968 ± 0.0042 (biraz daha düşük)

**ElasticNet**
- Ridge + Lasso karışımı.
- Sonuç: CV R² = 0.1084 ± 0.0069

#### AĞAÇ TABANLI MODELLER

**Decision Tree (Karar Ağacı)**
- Veriyi if/else kurallarıyla böler. Çok sezgisel ama overfitting'e aşırı eğilimlidir.
- Sonuç: CV R² = 0.0244 ± 0.0435 → **Kötü!** Hem düşük hem çok değişken (yüksek varyans).

**Random Forest (Rastgele Orman)**
- Yüzlerce karar ağacının ortalaması. Her ağaç rastgele feature alt kümesi kullanır (bagging).
- Sonuç: CV R² = 0.1584 ± 0.0225 → Güçlü baseline
- Tuning sonrası Test R² = 0.1307

**Gradient Boosting (Gradyan Artırma)**
- Ağaçlar sırayla eğitilir; her yeni ağaç öncekinin hatalarını düzeltir (boosting).
- Sonuç: CV R² = 0.1507 ± 0.0273
- Tuning sonrası Test R² = 0.1325 → **En iyi test skoru**

**XGBoost**
- Gelişmiş Gradient Boosting. Hızlı, regularizasyonlu.
- Sonuç: R² = 0.1068, MAE = 0.1468, RMSE = 0.1854

---

### 3.2 Model Seçimi — En İyi Model Hangisi?

**Seçilen model: Tuned Gradient Boosting**

| Model | CV R² | Test R² | Karar |
|-------|--------|---------|-------|
| Linear Regression | 0.1114 | — | Baseline |
| Ridge | 0.1114 | — | Baseline |
| Lasso | 0.0968 | — | Zayıf |
| ElasticNet | 0.1084 | — | Orta |
| Decision Tree | 0.0244 | — | ❌ Overfit |
| Random Forest (tuned) | **0.1769** | 0.1307 | Güçlü |
| **Gradient Boosting (tuned)** | 0.1719 | **0.1325** | ✅ **En iyi** |
| XGBoost | — | 0.1068 | İyi ama tuning az |

**Gradient Boosting neden seçildi?**
- Test seti R²'si en yüksek (0.1325)
- Random Forest'ın CV R²'si biraz daha yüksek (0.1769) ama test R²'si daha düşük → Random Forest biraz overfit etmiş
- Gradient Boosting genelleme (generalization) açısından daha tutarlı

---

### 3.3 Hiperparametre Ayarı (Hyperparameter Tuning)

#### Random Forest — Randomized Search
Denenen parametreler:
- `n_estimators` = 261 (ağaç sayısı)
- `max_depth` = 10 (her ağacın maksimum derinliği)
- `max_features` = sqrt (her bölünmede kaç feature denensin)
- `min_samples_leaf` = 2 (yaprak düğümlerde minimum örnek)
- `min_samples_split` = 13 (bölünme için minimum örnek)

Sonuç: CV R² 0.1584 → 0.1769 (iyileşme: +0.0185)

#### Gradient Boosting — Grid Search
En iyi parametreler:
- `learning_rate` = 0.05 (her adımda ne kadar öğrenilsin)
- `max_depth` = 3 (ağaç derinliği sınırı)
- `n_estimators` = 200 (kaç ağaç eklensin)
- `subsample` = 0.8 (her ağaç için verinin %80'i kullanılsın)

Sonuç: CV R² 0.1507 → 0.1719 (iyileşme: +0.0212)

---

### 3.4 Cross-Validation (Çapraz Doğrulama) — Fold Bazlı

**5-Fold CV nedir?**
Veri 5 eşit parçaya bölünür. Her seferinde 4 parça eğitim, 1 parça doğrulama için kullanılır. 5 kez tekrar edilir, ortalama alınır.

**Neden kullanırız?**
Tek bir eğitim/test bölünmesi şansa bağlı olabilir. CV, modelin gerçek performansını daha güvenilir ölçer.

| Model | Fold 1-5 Sonuçları | Ortalama | Std |
|-------|-------------------|----------|-----|
| Linear Reg. | 5 katlamada tutarlı | 0.1114 | ±0.0130 |
| Decision Tree | Büyük salınımlar | 0.0244 | **±0.0435** ← kötü |
| Random Forest | Stabil | 0.1769 | ±0.0225 |
| Gradient Boosting | Stabil | 0.1719 | ±0.0273 |

**Düşük std neden iyidir?** Modelin farklı veri dilimlerinde tutarlı davrandığını gösterir.

---

### 3.5 Metrikler — Ne Anlama Geliyor?

#### R² (R-Kare / Belirlilik Katsayısı)
- 0'dan 1'e kadar (negatif olabilir → çok kötü model demek)
- 1.0 = mükemmel tahmin
- 0.0 = modelin hiç bilgi üretmediği, sadece ortalamayı tahmin ettiği durum
- **Bizim değerimiz: 0.1325** → Modelin açıkladığı varyans %13.25

**Neden düşük R²?** Review skoru önceden tahmin edilmesi çok zor bir değişken. Oyunun gerçek kalitesi, pazarlama, topluluk etkileşimi gibi pek çok faktör bu özelliklerden ölçülemiyor.

#### RMSE (Root Mean Squared Error — Kök Ortalama Kare Hata)
- Tahmin hatalarının ortalama büyüklüğü (aynı birimde: 0–1 Wilson skalasında)
- XGBoost için RMSE = 0.1854 → Ortalama 0.185 birim hata

#### MAE (Mean Absolute Error — Ortalama Mutlak Hata)
- Hataların mutlak ortalaması. RMSE'den daha az uç değerlere duyarlı.
- XGBoost için MAE = 0.1468 → Ortalama 0.147 birim hata

**Pratikte ne demek?** Wilson skoru 0–1 arasında. 0.05 hata = 5 puanlık sapma (örn. %68 yerine %73 tahmin etmek).

---

### 3.6 Feature Importance (Özellik Önemi)

**Ne anlama gelir?**
Modelin tahmin yaparken hangi özelliklere ne kadar önem verdiğini gösterir.

**İki yöntem kullanıldı:**

**Tree-based importance (Ağaç tabanlı önem):**
- Her feature'ın ağaç bölünmelerinde ne kadar hata azalttığını ölçer
- Random Forest ve Gradient Boosting'de doğal olarak hesaplanır

**SHAP (SHapley Additive exPlanations):**
- Oyun teorisinden gelen yaklaşım
- Her tahmin için hangi feature'ın ne kadar katkı sağladığını hesaplar
- Daha adil ve güvenilir (tree-based importance bazı durumlarda yanıltıcı olabilir)

**Önemli feature'lar:** `dev_game_count`, `language_count`, `price_usd`, `release_year`, `achievement_count`

**`post_chatgpt` katsayısı:** Lineer modellerde +0.0058 (pozitif → ChatGPT sonrası oyunlar daha yüksek skor alıyor)

---

### 3.7 Residual Analizi (Artık Analizi)

**Residual nedir?**
Gerçek değer - Tahmin edilen değer = Residual (artık/hata)

**İyi bir modelde artıklar:**
- Sıfır çevresinde rastgele dağılmalı (sistematik hata yok)
- Normal dağılıma yakın olmalı (Jarque-Bera testi)
- Belirli bir yapı/örüntü göstermemeli

**Bizim sonuçlarımız:**
- Artıklar sıfır etrafında dağılmış ✓
- Jarque-Bera testi → Normale yakın ✓
- Belirgin bir sapma örüntüsü yok ✓

**Modelin hata yaptığı durumlar:**
- Aşırı düşük skoru olan oyunlar (0.0–0.4 arası) → Model bu uç değerleri tahmin etmekte zorlanıyor
- Aşırı yüksek skoru olan oyunlar (0.95+) → Viral/çok özel oyunlar için de zor
- Orta bölge (0.5–0.8) en iyi performans gösterilen alan

---

## BÖLÜM 4 — ZAMAN SERİSİ ANALİZİ

### 4.1 Bu Analizin Amacı Ne?

Regresyon modeli oyun bazlı tahmin yapıyor. Zaman serisi ise **aylık ortalama skorların** zaman içinde nasıl değiştiğini ve ChatGPT lansmanında bir kırılma yaşanıp yaşanmadığını test ediyor.

**Veri:** Ocak 2020 – Aralık 2024 arası 60 aylık gözlem
**Hedef:** `wilson_mean` (aylık ortalama Wilson skoru)

### 4.2 Temel Bulgular

**ChatGPT öncesi ortalama:** 0.6585
**ChatGPT sonrası ortalama:** 0.6978
**Fark:** +0.0394 (p < 0.0001) → İstatistiksel olarak anlamlı

Bu demek oluyor ki ChatGPT lansmanından sonra indie oyunların ortalama review skoru yaklaşık 4 puan arttı.

### 4.3 Yapısal Kırılma Testleri

#### Chow Testi
- Kasım 2022'de kırılma var mı? → **Evet** (F=3.81, p=0.0281)
- H0 reddedildi: Kasım 2022'de anlamlı yapısal kırılma ✓

#### CUSUM Testi (Önemli!)
- En güçlü kırılma noktası: **Ocak 2022** (ChatGPT'den 10 ay önce!)
- Bu şunu gösteriyor: Kalite artışı ChatGPT'den önce başlamış
- Olası açıklama: Midjourney (Eylül 2021), GitHub Copilot (Ekim 2022) gibi araçlar piyasayı daha önce etkiledi

#### Mann-Whitney U Testi
- Etki büyüklüğü: r = 0.704 → **Büyük etki** ✓
- p < 0.0001 → Son derece anlamlı

### 4.4 Zaman Serisi Modelleri

#### ARIMA(1,1,0)
- AR(1): Bir önceki ayın değeri tahmin için kullanılıyor
- d=1: Bir fark alındı (durağanlık sağlandı)
- MA(0): Hareketli ortalama bileşeni yok

**Test Seti Performansı:**
- MAPE: **0.86%** → Mükemmel! (%10'un çok altında)
- MAE: 0.0062
- RMSE: 0.0077

#### Naif Baseline (Son değeri kopyala)
- MAPE: 0.86% → ARIMA ile **aynı!**
- Bu ilginç bir bulgu: Serinin yapısı o kadar basit ki, "bu ay geçen aya eşit" demek ARIMA kadar iyi

#### Prophet
- MAPE: 2.94% (ARIMA'dan kötü ama kırılma noktası tespitinde üstün)
- Changepoint detection ile ChatGPT lansmanını otomatik tespit ediyor

---

## BÖLÜM 5 — CANLÜ DEMO (5 Dakika)

### Demo Nasıl Çalıştırılır?

```bash
cd /Users/ahmetboz/developer/bozappz/chatgpt-impact-indie-games
python demo_server.py
# Sonra tarayıcıda: http://localhost:5000
```

### Demo Sırasında Denenebilecek Kombinasyonlar

**Kombinasyon 1 — "Tipik Başarılı Indie Oyun":**
```
dev_game_count = 3
solo_dev_proxy = 0
dev_equals_publisher = 1
achievement_count = 25
has_demo = 1
dlc_count = 2
genre_count = 3
platform_count = 2
language_count = 10
price_usd = 14.99
is_early_access = 0
post_chatgpt = 1  ← ChatGPT sonrası
release_year = 2024
has_workshop = 0
```
Beklenen skor: ~0.65–0.72

**Kombinasyon 2 — "ChatGPT Öncesi Benzer Oyun":**
Yukarıdakiyle aynı ama `post_chatgpt = 0` ve `release_year = 2021`
Beklenti: Biraz daha düşük skor (fark küçük, ~0.006)

**Kombinasyon 3 — "Solo Dev, Düşük Bütçe":**
```
dev_game_count = 1
solo_dev_proxy = 1
dev_equals_publisher = 1
achievement_count = 5
has_demo = 0
dlc_count = 0
genre_count = 2
platform_count = 1
language_count = 3
price_usd = 4.99
is_early_access = 1
post_chatgpt = 1
release_year = 2024
has_workshop = 0
```
Beklenti: Daha düşük skor (~0.55–0.62)

**Kombinasyon 4 — "Büyük Stüdyo Tarzı Indie":**
```
dev_game_count = 15
solo_dev_proxy = 0
dev_equals_publisher = 0
achievement_count = 100
has_demo = 1
dlc_count = 10
genre_count = 4
platform_count = 3
language_count = 20
price_usd = 29.99
is_early_access = 0
post_chatgpt = 1
release_year = 2024
has_workshop = 1
```
Beklenti: Yüksek skor (~0.70–0.78)

### Demo'da Ne Söylersin?

1. "Bu arayüzde modele oyun özellikleri giriyoruz ve model bize tahmin edilen review skoru veriyor."
2. İlk kombinasyonu gir, skoru göster.
3. Sadece `post_chatgpt = 0` yap, farkı göster → "ChatGPT sonrası olması modele göre skoru ~0.006 artırıyor."
4. "Bu fark küçük görünüyor ama zaman serisi analizimiz toplam popülasyonda +0.039'luk bir artış gösteriyor."
5. Feature importance grafiğini göster (varsa): "Modele göre en önemli feature'lar şunlar..."

---

## BÖLÜM 6 — KAVRAM SÖZLÜĞÜ (Tüm Terimler Türkçe)

### Temel İstatistik Terimleri

| Terim | Türkçe Açıklama |
|-------|-----------------|
| **Regression (Regresyon)** | Sayısal bir çıktıyı tahmin etmeye çalışan model türü. "Bu oyun kaç puan alır?" gibi sorular. |
| **Classification (Sınıflandırma)** | Kategorik çıktı tahmin eder. "Bu oyun başarılı mı, değil mi?" |
| **Feature (Özellik/Girdi)** | Modele verilen bilgi parçaları. Örn: fiyat, dil sayısı. |
| **Target/Label (Hedef Değişken)** | Tahmin etmeye çalıştığımız değer. Bizde: review skoru. |
| **Train/Test Split (Eğitim/Test Ayrımı)** | Veriyi ikiye böl: Modeli bir kısmıyla eğit, diğeriyle test et. |
| **Overfitting (Aşırı Öğrenme)** | Model eğitim verisini ezberliyor, yeni veriyle başarısız. |
| **Underfitting (Yetersiz Öğrenme)** | Model çok basit, veriyi öğrenemiyor. |
| **Bias-Variance Tradeoff** | Basit model → yüksek bias (hatalı). Karmaşık model → yüksek varyans (kararsız). İkisinin dengesi. |
| **Generalization (Genelleme)** | Modelin eğitimde görmediği veriye ne kadar iyi uyum sağladığı. |

### Model Metrikleri

| Terim | Türkçe Açıklama |
|-------|-----------------|
| **R² (R-kare)** | Modelin hedef değişkendeki varyansı ne kadar açıkladığı. 1.0 = mükemmel, 0.0 = hiç açıklamıyor. |
| **RMSE** | Kök Ortalama Kare Hata. Hataları kareler alıp ortalayıp köke çekiyoruz. Büyük hatalar daha çok cezalandırılır. |
| **MAE** | Ortalama Mutlak Hata. Hataların mutlak değerlerinin ortalaması. RMSE'den daha yorumlanabilir. |
| **MAPE** | Ortalama Mutlak Yüzde Hata. %X hata yapıyoruz demek. %10 altı iyi kabul edilir. |
| **Residual (Artık)** | Gerçek - Tahmin. İyi modelde sıfır etrafında rastgele. |

### Model Türleri

| Terim | Türkçe Açıklama |
|-------|-----------------|
| **Linear Regression** | En basit regresyon. Y = a1*X1 + a2*X2 + ... + b |
| **Ridge (L2)** | Linear reg + büyük katsayılara ceza. Overfitting azaltır. |
| **Lasso (L1)** | Linear reg + gereksiz feature'ları sıfıra iter. Feature seçimi yapar. |
| **ElasticNet** | Ridge + Lasso karışımı. |
| **Decision Tree** | If/else kurallarla veriyi bölen ağaç yapısı. Yorumlanması kolay ama overfit riski yüksek. |
| **Random Forest** | Çok sayıda karar ağacının ortalaması. Bagging yöntemi. |
| **Gradient Boosting** | Ağaçlar sırayla eklenir; her yeni ağaç öncekinin hatalarını düzeltir. Boosting yöntemi. |
| **XGBoost** | Optimize edilmiş Gradient Boosting. Hız ve regularizasyon ekler. |
| **ARIMA** | Zaman serisi modeli. Geçmiş değerler + hatalar kullanılır. |
| **Prophet** | Facebook'un zaman serisi modeli. Mevsimsellik ve kırılma noktalarını otomatik bulur. |

### Cross-Validation Terimleri

| Terim | Türkçe Açıklama |
|-------|-----------------|
| **Cross-Validation (CV)** | Çapraz Doğrulama. Veriyi birden fazla şekilde bölerek model performansını daha güvenilir ölç. |
| **K-Fold CV** | Veri K parçaya bölünür. K kez eğit-test döngüsü yapılır. Biz K=5 kullandık. |
| **Fold** | CV'deki her bir test dilimi. 5-fold'da 5 ayrı test diliminiz var. |
| **CV Score ± Std** | Ortalama CV skoru ± standart sapma. Std küçükse model stabil. |

### Hiperparametre Terimleri

| Terim | Türkçe Açıklama |
|-------|-----------------|
| **Hyperparameter (Hiperparametre)** | Model eğitilmeden önce belirlenen ayarlar. Örn: ağaç sayısı. |
| **Grid Search** | Tüm parametre kombinasyonlarını dene, en iyisini bul. Yavaş ama kapsamlı. |
| **Randomized Search** | Rastgele kombinasyonlar dene. Grid search'ten hızlı. |
| **n_estimators** | Random Forest/GB'de kaç ağaç olsun. |
| **max_depth** | Ağacın maksimum derinliği. Düşükse underfitting, yüksekse overfitting. |
| **learning_rate** | GB'de her adımda ne kadar öğrenilsin. Düşük = stabil ama yavaş. |
| **subsample** | GB'de her ağaç için verinin kaçta biri kullanılsın (%80 → 0.8). |

### Feature Importance Terimleri

| Terim | Türkçe Açıklama |
|-------|-----------------|
| **Feature Importance** | Özellik Önemi. Hangi girdi modelin tahminini ne kadar etkiliyor. |
| **SHAP Values** | Her tahmin için her feature'ın katkısını oyun teorisiyle hesaplar. Daha adil ve açıklanabilir. |
| **Mean Decrease Impurity** | Ağaç tabanlı önem hesabı. Feature kaç kez ve ne kadar hata azaltarak bölündü. |
| **Permutation Importance** | Feature değerlerini karıştır, performans düşüyorsa o feature önemli. |

### Zaman Serisi Terimleri

| Terim | Türkçe Açıklama |
|-------|-----------------|
| **Stationarity (Durağanlık)** | Serinin istatistiksel özellikleri (ortalama, varyans) zamanla değişmiyorsa durağan. ARIMA durağan seri ister. |
| **ADF Testi** | Augmented Dickey-Fuller. Durağanlık testi. p < 0.05 → durağan. |
| **Differencing (Fark Alma)** | t anındaki değerden t-1 anındaki değeri çıkarma. Seriyi durağan hale getirir. |
| **ACF** | Autocorrelation Function. Serinin kendisiyle farklı gecikmelerde korelasyonu. |
| **PACF** | Partial ACF. Ara gecikmelerin etkisi kontrol edilerek hesaplanan korelasyon. |
| **Lag (Gecikme)** | Geçmiş zaman adımı. Lag-1 = bir önceki ay. |
| **Rolling Window** | Kayan pencere. Son N ayın ortalaması/std gibi özellikler üretir. |
| **Structural Break (Yapısal Kırılma)** | Zaman serisinde ani bir değişim/sıçrama. |
| **Chow Testi** | Belirli bir noktada yapısal kırılma olup olmadığını test eder. |
| **CUSUM** | Cumulative Sum. Kümülatif hata takibi ile kırılma noktası bulur. |
| **Mann-Whitney U** | İki bağımsız grubun dağılımını karşılaştıran parametrik olmayan test. |
| **Changepoint** | Prophet'in otomatik tespit ettiği değişim noktası. |
| **Seasonality (Mevsimsellik)** | Belirli periyotlarda tekrar eden örüntü (her yaz artış gibi). |
| **Trend** | Uzun vadeli yükseliş veya düşüş yönü. |
| **MAPE** | Mean Absolute Percentage Error. %X hata → %10 altı kabul edilebilir. |

### Wilson Lower Bound

| Terim | Türkçe Açıklama |
|-------|-----------------|
| **Wilson Lower Bound** | İstatistiksel güvene dayalı review skoru. Az yorumlu oyunları cezalandırır. %90 pozitif ama 10 yorum = düşük skor. %85 pozitif ama 10.000 yorum = yüksek skor. |
| **Confidence Interval** | Güven aralığı. "Gerçek değer %95 ihtimalle bu aralıkta." |

---

## BÖLÜM 7 — HOCANIN MUHTEMELen SORACAĞI SORULAR VE CEVAPLAR

### S1: "R² değeriniz sadece 0.13. Bu çok düşük değil mi? Modeliniz işe yaramıyor mu?"

**Cevap:**
"0.13 düşük görünüyor ama sosyal veri için bu makul bir değer. Review skoru oyunun gerçek kalitesine, pazarlama bütçesine, topluluk etkileşimine bağlı. Bunların hiçbirini veri setimizden ölçemiyoruz. Pre-release (çıkış öncesi) özelliklerle sadece %13 varyansı açıklamak aslında bu domain'in sınırlılığını gösteriyor. Önemli olan modelin istatistiksel olarak anlamlı örüntüler yakaladığı ve ChatGPT etkisini diğer faktörler kontrol altında tutarak ölçebildiğimiz."

### S2: "Neden Gradient Boosting'i seçtiniz, Random Forest daha yüksek CV skoru almıyor mu?"

**Cevap:**
"Random Forest'ın CV R²'si 0.1769, Gradient Boosting'in 0.1719 — RF biraz önde. Ama test setinde RF 0.1307, GB 0.1325 alıyor. Yani RF eğitim verisi üzerinde biraz daha iyi görünüyor ama görmediği test verisinde GB'nin gerisinde. Bu, RF'ın hafifçe overfit ettiğine işaret ediyor. Test seti görülmemiş veriyi temsil ettiğinden GB'yi seçtik. Ama ikisi arasındaki fark çok küçük — her iki model de savunulabilir."

### S3: "Decision Tree neden bu kadar kötü?"

**Cevap:**
"Decision Tree CV R² 0.0244 ve standart sapması ±0.0435 — çok yüksek varyans. Yani farklı veri dilimlerine göre performansı dramatik değişiyor. Bunun sebebi: pruning (budama) yapılmadığında ağaç eğitim verisini ezberliyor. 5-fold CV'de bu ezberleme farklı katlarda çalışmıyor. Bu tam olarak ensemble yöntemlerin (Random Forest: çok ağaç ortalaması) Decision Tree'ye neden üstün olduğunu gösteriyor."

### S4: "Yapısal kırılma ChatGPT'den 10 ay önce Ocak 2022'de neden?"

**Cevap:**
"Bu ilginç bir bulgu. CUSUM testi en güçlü kırılmayı Ocak 2022'de buluyor. Olası açıklamalar: Midjourney Eylül 2021'de, GitHub Copilot Ekim 2022'de piyasaya çıktı. AI araçlarının geliştiriciler üzerindeki etkisi ChatGPT'den önce başlamış olabilir. Ayrıca ChatGPT lansmanı piyasada beklenti yarattı — geliştiriciler hazırlanmaya başlamış olabilir. Chow testi Kasım 2022'de de anlamlı bir kırılma buluyor; bu ChatGPT lansmanını doğruluyor. Yani kırılma gerçek ama ChatGPT tek neden olmayabilir — daha geniş bir AI dalgasının parçası."

### S5: "ARIMA ile naif baseline aynı performansı gösteriyor. ARIMA'nın ne faydası var?"

**Cevap:**
"Harika bir soru. Naif baseline 'bu ay geçen aya eşit' demek. ARIMA da AR(1) yapısıyla temelde aynı şeyi yapıyor — lag-1 korelasyonu 0.506 çok güçlü, yani geçen ay bu ayı güçlü açıklıyor. Bu aslında serinin yapısını ortaya koyuyor: aylık review kalitesi bir önceki aydan güçlü etkileniyor, ötesinde çok fazla eklenecek bilgi yok. ARIMA'nın avantajı: (1) İstatistiksel varsayımları test edebildik (white noise, normallik), (2) Belirsizlik aralıkları üretiyor, (3) Ölçeklenebilir — daha uzun serilerde naif baseline'ı geçer."

### S6: "Modelinizin hangi değer aralıklarında hata yaptığını söyleyin."

**Cevap:**
"Residual analizinde iki uç bölgede hataların arttığını gördük: (1) Çok düşük skora sahip oyunlar (Wilson 0.0–0.4): Bunlar genellikle çok az yorum alan veya gerçekten kötü oyunlar. Pre-release özellikler bu başarısızlığı tahmin etmekte yetersiz. (2) Çok yüksek skora sahip oyunlar (0.95+): Viral olan, beklenmedik başarılar. Bunları önceden tahmin etmek neredeyse imkansız. En doğru tahmin orta bölgede (0.5–0.8) yapılıyor — oyunların çoğunluğu zaten bu bölgede."

### S7: "Post_chatgpt feature'ı doğrudan hedefinizde var mı? Bu data leakage değil mi?"

**Cevap:**
"Hayır, bu data leakage değil. `post_chatgpt` çıkış zamanını kodluyor — yayın öncesinde bilinen bir bilgi. Oyun piyasaya çıkmadan önce ne zaman çıkacağını (dolayısıyla ChatGPT sonrası mı öncesi mi) biliyoruz. Leakage olsaydı, hedeften elde edilen ya da geleceğe ait bir bilgi kullanıyor olurduk."

### S8: "Neden log(review_count) veya başka transform denediniz mi?"

**Cevap:**
"Wilson lower bound zaten bir tür dönüşüm — ham positive rate'in güvene dayalı versiyonu. Hedef değişkende ek log transform denemedik; Wilson skoru [0,1] arasında sınırlı ve görece normal dağılımlı. Hedef dönüşümü bazen RMSE optimizasyonunu bozabilir. Söyleyebileceğimiz sınırlılık: feature'lar için log transform (özellikle dev_game_count gibi sağa çarpık olanlar için) denenebilirdi."

### S9: "Prophet neden daha kötü ama yine de neden bahsettiniz?"

**Cevap:**
"Test MAPE açısından: ARIMA %0.86, Prophet %2.94. Prophet daha kötü ama yanlış soruyu yanıtlıyor aslında. Biz sadece tahmin doğruluğu istemiyoruz — kırılma noktasını anlamak istiyoruz. Prophet'in changepoint detection'ı 'ne zaman değişti?' sorusunu otomatik ve görsel olarak yanıtlıyor. Bu, araştırma sorumuza daha direkt cevap veriyor. Stakeholder'lara (hoca, panel) gösterirken Prophet çok daha anlaşılır grafik üretiyor."

### S10: "Modeliniz gerçek hayatta kullanılabilir mi?"

**Cevap:**
"Üç sınırlılığımız var: (1) R² = 0.13 — yani varyansın %87'si açıklanamıyor. Kesin tahmin için yetersiz. (2) Veriler 2020-2024 arası; piyasa değiştikçe model yeniden eğitilmeli. (3) Sadece Steam indie oyunları — AAA veya mobil oyunlara genellenemez. Kullanım alanı: Oyun geliştiricisi 'çıkaracağım oyunun review skoru ne olabilir?' diye bir fikir edinmek istiyorsa bu model kabaca yön verebilir. Yatırım kararı vermek için yetersiz."

---

## BÖLÜM 8 — SUNUM AKIŞI (Script)

### Açılış (30 saniye)
"Bu sunum, önceki EDA sunumumuzun devamıdır. Araştırma sorumuzu kısaca hatırlatmak istiyorum: **ChatGPT'nin piyasaya sürülmesi Steam indie oyunlarının review skorlarını sistematik olarak değiştirdi mi?** Bu soruyu iki yöntemle test ettik: regresyon modeli ve zaman serisi analizi."

### Demo (5 dakika)
"Önce canlı demomuzu gösterelim. [demo_server.py çalıştır, tarayıcı aç] Bu arayüzde pre-release özellikler giriyoruz ve model tahmin üretiyor. [Kombinasyon 1 gir] Bu tipik bir başarılı indie oyun profili... [skoru göster]. Şimdi sadece ChatGPT sonrası bayrağını değiştirelim [post_chatgpt = 0 yap]... Fark küçük, yaklaşık 0.006. Ama bunu zaman serisi sonuçlarıyla birleştirince anlamlı bir tablo ortaya çıkıyor."

### Model Kararları (10 dakika)
"8 farklı model denedik. [Tablo göster] Başlangıçta doğrusal modeller, sonra ağaç tabanlılar. Decision Tree ciddi overfit gösterdi — CV standart sapması ±0.0435. Ensemble modeller öne çıktı. 5-fold cross-validation ile Gradient Boosting ve Random Forest yarıştırdık. Hiperparametre tuning sonrası Gradient Boosting 0.1325 test R²'si ile seçildik — Random Forest'tan biraz daha iyi genelleme yapıyor. Feature importance açısından: [en önemli feature'ları say]. `post_chatgpt` pozitif katsayı (+0.0058) taşıyor."

### Sonuç (5 dakika)
"Modelimizin güçlü yönleri: stabil, overfitting yok, ChatGPT etkisini istatistiksel kontrol altında ölçüyor. Zayıf yönleri: R² 0.13 — görülmemiş faktörler baskın. Uç değerli oyunlarda hata artıyor. Sınırlılıklar: sadece Steam indie, 2020-2024 verisi. Ana bulgu: ChatGPT lansmanından sonra indie oyun review kalitesi ortalama ~4 puan arttı ve bu fark istatistiksel olarak anlamlı. Ancak CUSUM analizi kırılmanın ChatGPT'den 10 ay önce başladığını gösteriyor — bu daha geniş bir AI dalgasının etkisi olabilir."

---

## BÖLÜM 9 — RAKAMLAR (Ezberle)

```
Veri seti:         4.363 oyun, 14 feature, 80/20 split
En iyi model:      Tuned Gradient Boosting
Test R²:           0.1325
CV R²:             0.1719 ± 0.0273
RF CV R²:          0.1769 (biraz daha iyi ama test'te geride)
RF Test R²:        0.1307

ChatGPT öncesi ortalama:  0.6585
ChatGPT sonrası ortalama: 0.6978
Fark:                     +0.0394 (p < 0.0001)

ARIMA MAPE:        0.86% (hedef < %10 ✓)
Prophet MAPE:      2.94%
Zaman serisi:      60 aylık gözlem (Ocak 2020 – Aralık 2024)

Yapısal kırılma:
  Chow Testi:     Kasım 2022'de anlamlı (p=0.0281) ✓
  CUSUM:          Ocak 2022'de daha güçlü (p=0.0021) ← 10 ay erken
  Mann-Whitney:   r = 0.704 büyük etki boyutu
```

---

## BÖLÜM 10 — MODELI ÇALIŞTIRMA TALİMATI (Demo Öncesi Hazırlık)

```bash
# 1. Gerekli kütüphaneler yüklü mu?
pip install flask flask-cors joblib scikit-learn numpy pandas

# 2. Model dosyası var mı? Kontrol et:
ls /Users/ahmetboz/developer/bozappz/chatgpt-impact-indie-games/model.joblib

# 3. Sunucuyu başlat:
cd /Users/ahmetboz/developer/bozappz/chatgpt-impact-indie-games
python demo_server.py

# 4. Terminalde şunu görmelisin:
# [demo_server] Loading model from: model.joblib
# [demo_server] Model loaded: RandomForestRegressor (veya Pipeline)
# Open in browser: http://localhost:5000

# 5. Tarayıcıda aç:
# http://localhost:5000
```

**Önceden test et! Sunum günü sürpriz istemezsin.**

---

*Bu rehber ML_Regression_Extended.ipynb, ML_TimeSeries_Extended.ipynb ve demo_server.py dosyaları analiz edilerek hazırlanmıştır.*
