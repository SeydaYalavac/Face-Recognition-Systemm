# access_control.py
try:
    from plyer import vibrator, tts, notification
    PLYER_HAZIR = True
except ImportError:
    PLYER_HAZIR = False

import json
import os
import datetime

# AYARLAR
LOG_DOSYASI = "erisim_log.json"
MAX_HATALI_GIRIS = 3
GUVEN_ESIGI = 75.0

def donanim_tetikle(durum, mesaj):
    if not PLYER_HAZIR:
        print(f"[SIMULASYON]: Donanim -> {durum}")
        return
    try:
        if durum == "onay":
            tts.speak("Erisim onaylandi. Hos geldiniz.")
            vibrator.vibrate(time=0.2)
        elif durum == "red":
            tts.speak("Erisim reddedildi.")
            vibrator.vibrate(time=0.5)
            notification.notify(title="Guvenlik Uyarisi", message=mesaj)
        elif durum == "alarm":
            tts.speak("Alarm aktif edildi. Guvenlik ihlali.")
            vibrator.vibrate(time=1.0)
            notification.notify(title="ALARM", message=mesaj)
    except:
        pass

def log_kaydet(kayit_verisi):
    loglar = []
    if os.path.exists(LOG_DOSYASI):
        with open(LOG_DOSYASI, "r", encoding="utf-8") as f:
            try:
                loglar = json.load(f)
            except:
                loglar = []
    loglar.append(kayit_verisi)
    with open(LOG_DOSYASI, "w", encoding="utf-8") as f:
        json.dump(loglar, f, ensure_ascii=False, indent=2)

def erisim_karari_uret(yuz_verisi, bellek_interface, user_id):
    """
    [span_0](start_span)Kisi 2'nin Kisi3Interface yapisini kullanarak karar verir[span_0](end_span).
    """
    # [span_1](start_span)Kisi 2'nin arayuzunden veriyi resmi yolla cekiyoruz[span_1](end_span)
    bellek_verisi = bellek_interface.get_user_data(user_id)
    
    sonuc = {
        "access_granted": False,
        "message": "Erisim Reddedildi",
        "alarm": 0,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # 1. [span_2](start_span)KONTROL: Kullanici kaydi bellekte var mi?[span_2](end_span)
    if not bellek_verisi.get("found", False):
        sonuc["message"] = "HATA: Kullanici kaydi bulunamadi!"
        donanim_tetikle("red", sonuc["message"])
        return sonuc

    # 2. KONTROL: Yuz taninma ve Guven Skoru (Kisi 1 Verisi)
    if not yuz_verisi.get("recognized", False) or yuz_verisi.get("confidence_score", 0) < GUVEN_ESIGI:
        # [span_3](start_span)Hatali girisi Kisi 2'nin bellek metoduna bildir[span_3](end_span)
        bellek_interface.record_failure(user_id)
        
        yeni_hata_sayisi = bellek_verisi.get('failed_attempts', 0) + 1
        sonuc["message"] = f"DUSUK GUVEN VEYA HATALI YUZ! (Hata: {yeni_hata_sayisi}/{MAX_HATALI_GIRIS})"
        donanim_tetikle("red", sonuc["message"])
        
        # Alarm Kontrolu: Hata sayisi sinira ulasti mi?
        if yeni_hata_sayisi >= MAX_HATALI_GIRIS:
            sonuc["alarm"] = 1
            sonuc["message"] = "!!! ALARM AKTIF: COKLU HATALI GIRIS !!!"
            donanim_tetikle("alarm", sonuc["message"])
            
    # 3. [span_4](start_span)KONTROL: Kullanici Durumu (Status) ve Yetki (Authority)[span_4](end_span)
    elif bellek_verisi.get("status") == 1 and bellek_verisi.get("authority_level", 0) >= 1:
        # [span_5](start_span)Basarili girisi Kisi 2'nin bellek metoduna bildir[span_5](end_span)
        bellek_interface.record_success(user_id)
        sonuc["access_granted"] = True
        sonuc["message"] = f"Erisim Onaylandi. Hos geldiniz, {user_id}."
        donanim_tetikle("onay", "Giris Basarili")
    
    else:
        sonuc["message"] = "Yetkisiz veya Pasif Kullanici Durumu."
        donanim_tetikle("red", sonuc["message"])

    # Log Kayit islemi
    log_kaydet({**yuz_verisi, **sonuc, "auth_level": bellek_verisi.get("authority_level")})
    return sonuc

def log_goruntule():
    if os.path.exists(LOG_DOSYASI):
        with open(LOG_DOSYASI, "r", encoding="utf-8") as f:
            try:
                loglar = json.load(f)
                print("\n" + "="*55)
                print("       --- SISTEM ERISIM KAYITLARI (LOGLAR) ---")
                print("="*55)
                for l in loglar:
                    z = l.get("timestamp", "N/A")
                    n = l.get("name") or l.get("user_id") or "Bilinmiyor"
                    o = "ONAY" if l.get("access_granted") else "RED"
                    a = " [ALARM!]" if l.get("alarm") else ""
                    print(f"[{z}] {n} -> {o}{a}")
                print("="*55 + "\n")
            except:
                pass

# --- TEST SENARYOSU ---
if __name__ == "__main__":
    class MockInterface:
        def get_user_data(self, uid): 
            return {"found": True, "authority_level": 2, "status": 1, "failed_attempts": 0}
        def record_success(self, uid): print(f"Sistem: {uid} basarili giris kaydedildi.")
        def record_failure(self, uid): print(f"Sistem: {uid} hatali giris kaydedildi.")

    interface = MockInterface()
    yuz_test = {"user_id": "ID_002", "name": "sevde", "recognized": True, "confidence_score": 88.0, "pose": "front"}
    
    print("\n--- KISI 2 ENTEGRASYON TESTI ---")
    final_sonuc = erisim_karari_uret(yuz_test, interface, "ID_002")
    print(f"KARAR: {final_sonuc['message']}")
    log_goruntule()