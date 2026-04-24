# TODO - Guven Skoru 0 Sorunu Duzeltme

## Plan
- [x] Analiz tamamlandi
- [x] `config.py` esik degerlerini duzelt
- [x] `recognize_user.py` `conf_to_display_score()` mapping'ini duzelt
- [x] `recognize_user.py` `avg_score` hesaplamasini duzelt
- [x] Test ve kontrol

## Yapilan Degisiklikler

### config.py
- `LBPH_STRICT_THRESHOLD`: 70.0 → 75.0
- `LBPH_SOFT_THRESHOLD`: 105.0 → 95.0
- `MIN_DISPLAY_SCORE_TO_ACCEPT`: 65 → 55

### recognize_user.py
- `conf_to_display_score()` mapping guncellendi. Yeni eşiklerle uyumlu daha doğrusal skorlar:
  - conf <= 40 → 98, conf <= 50 → 94, conf <= 60 → 90, conf <= 70 → 85
  - conf <= 75 → 80, conf <= 80 → 75, conf <= 90 → 68, conf <= 95 → 60
  - conf <= 100 → 55, conf <= 110 → 45, conf <= 120 → 35, conf <= 140 → 25, else → 15
- `avg_score` hesaplamasi duzeltildi: `valid_scores` bossa 0 yerine son non-zero skor korunuyor.

## Beklenen Sonuc
- Artik LBPH mesafesi 95 altinda olan tahminler `soft` olarak kabul edilecek.
- `conf_to_display_score(95)` = 60, `MIN_DISPLAY_SCORE_TO_ACCEPT(55)`'i geciyor.
- VIP kullanicilar icin esik 45'e dusuyor (55 - 10).
- Gecerli tahmin yoksa bile ekranda 0 yerine son gecerli skor gorunecek.

