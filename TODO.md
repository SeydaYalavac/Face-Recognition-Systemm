# TODO - Yuz Tanima Test Modu ve Guven Skoru Duzeltme

## Plan
- [x] Analiz tamamlandi
- [x] `config.py`'ye TEST_MODE bayragi ve test sabitleri ekle
- [x] `recognize_user.py` `conf_to_display_score()` mapping'ini duzelt (LBPH ters oranti)
- [x] `recognize_user.py`'ye users.json <-> trainer.yml label eslesme debug satiri ekle
- [x] `recognize_user.py` accept mantigini duzelt (finally blogundan cikar, dongu icine al)
- [x] `recognize_user.py`'ye Test Modu ekle (bulaniklik/parlaklik filtrelerini devre disi birak)
- [x] Syntax kontrolu ve test

## Beklenen Sonuc
- LBPH mesafesi dusukse guven skoru yuksek cikacak (`100 - conf`).
- `trainer.yml` label'lari ile `users.json` `numeric_id` degerleri konsolda gorunecek.
- Test modunda bulaniklik ve parlaklik filtreleri calismayacak.
- `accept` mantigi dogru calisacak, tanima basarili olacak.

## Not
- `config.py` icinde `TEST_MODE = True` olarak ayarlandi. Testler bittiginde `False` yapilmali.


