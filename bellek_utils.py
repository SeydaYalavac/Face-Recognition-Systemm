# bellek_utils.py
"""
Bellek işleme fonksiyonları
Kişi 1'den gelen veriyi belleğe yazma ve bellekten okuma
"""

import json
from datetime import datetime
from pathlib import Path
from config import (
    MEMORY_OFFSET, AUTHORITY_LEVELS, USER_STATUS, ALARM_STATUS,
    BASE_ADDRESS, ADDRESS_INCREMENT, BLOCK_SIZE, MAX_USERS,
    USERS_MEMORY_FILE, MEMORY_DUMP_FILE, LOG_FORMAT, LOG_ENABLED,
    MIN_CONFIDENCE_SCORE
)


class BellekSistemi:
    """
    Bellek Sistemi Yöneticisi
    Kişi 1'den gelen verileri işleyip bellekte saklayan sistem
    """
    
    def __init__(self):
        """Bellek sistemini başlat"""
        # Sanal bellek (adres -> veri)
        self.memory = {}
        
        # Kullanıcı indeksi (user_id -> base_address)
        self.user_index = {}
        
        # Kullanıcı bilgileri
        self.users = {}
        
        # İşlem logları
        self.logs = []
        
        # Aktif kullanıcı sayısı
        self.active_users_count = 0
        
        self._log("SISTEM", "Bellek Sistemi başlatıldı")
        self._log("SİSTEM", f"Max kapasite: {MAX_USERS} kullanıcı, Blok boyutu: {BLOCK_SIZE} alan")
        
        # Önceki durumu yükle
        self._load_state()
    
    # ============= ADRES İŞLEMLERİ =============
    
    def _calculate_base_address(self, user_number):
        """Kullanıcı numarasından base adresini hesapla"""
        return BASE_ADDRESS + (user_number * ADDRESS_INCREMENT)
    
    def _find_empty_block(self):
        """İlk boş bellek bloğunu bul"""
        for i in range(MAX_USERS):
            base_addr = self._calculate_base_address(i)
            if base_addr not in self.memory:
                return i, base_addr
        return None, None
    
    def _find_user_address(self, user_id):
        """user_id'ye ait base adresi bul"""
        return self.user_index.get(user_id)
    
    # ============= KİŞİ 1'DEN VERİ ALMA VE İŞLEME =============
    
    def kisi1_verisini_isle(self, kisi1_data):
        """
        Kişi 1'den gelen veriyi işle
        
        Args:
            kisi1_data (dict): Kişi 1'in çıktı verisi
                {
                    "camera_status": "aktif",
                    "face_detection": "basarili",
                    "landmark_analysis": "tamamlandi",
                    "pose": "front",
                    "recognized_user": "seyda",
                    "user_id": "ID_001",
                    "confidence_score": 82.0,
                    "recognized": true,
                    "next_step": "bellek kontrolune gonderildi"
                }
        
        Returns:
            dict: İşleme sonucu
        """
        
        # Veri doğrulama
        if not self._validate_kisi1_data(kisi1_data):
            return {
                "success": False,
                "error": "Kişi 1 verisi geçersiz",
                "data": kisi1_data
            }
        
        # Kamera kapalıysa işleme yapma
        if kisi1_data["camera_status"] != "aktif":
            self._log("UYARI", "Kamera aktif değil")
            return {
                "success": False,
                "error": "Kamera kapalı",
                "access_granted": False
            }
        
        # Yüz algılanamadıysa
        if kisi1_data["face_detection"] != "basarili":
            self._log("UYARI", "Yüz algılanamadı")
            return {
                "success": False,
                "error": "Yüz algılanamadı",
                "access_granted": False
            }
        
        # Tanıma başarısız oldu
        if not kisi1_data["recognized"]:
            self._log("UYARI", f"Tanıma başarısız: {kisi1_data['next_step']}")
            return {
                "success": False,
                "error": kisi1_data["next_step"],
                "access_granted": False
            }
        
        # Güven skoru düşük
        if kisi1_data["confidence_score"] < MIN_CONFIDENCE_SCORE:
            self._log("UYARI", f"Güven skoru düşük: {kisi1_data['confidence_score']}")
            return {
                "success": False,
                "error": f"Güven skoru yetersiz ({kisi1_data['confidence_score']}%)",
                "access_granted": False
            }
        
        # ✓ Tanıma başarılı
        user_id = kisi1_data["user_id"]
        username = kisi1_data["recognized_user"]
        confidence = kisi1_data["confidence_score"]
        
        self._log("BASARILI", f"Tanıma başarılı: {username} ({user_id}, Güven: {confidence}%)")
        
        # Kullanıcı bellekte var mı kontrol et
        user_info = self.kullanici_bilgisi_oku(user_id)
        
        if user_info is None:
            # Yeni kullanıcı ekleme
            success, result = self.kullanici_ekle(user_id, username)
            if not success:
                return {
                    "success": False,
                    "error": "Kullanıcı belleğe eklenemedi",
                    "access_granted": False
                }
        
        # Son erişim zamanını güncelle
        self.son_erisim_kaydet(user_id)
        
        # Başarılı giriş - hata sayısını sıfırla
        self.hata_sayisini_sifirla(user_id)
        self.alarm_kapat(user_id)
        
        # Sonuç döndür
        result = {
            "success": True,
            "user_id": user_id,
            "username": username,
            "confidence": confidence,
            "access_granted": True,
            "kisi1_data": kisi1_data
        }
        
        return result
    
    def _validate_kisi1_data(self, data):
        """Kişi 1 verisini doğrula"""
        required_fields = [
            "camera_status", "face_detection", "landmark_analysis",
            "pose", "recognized_user", "user_id", "confidence_score",
            "recognized", "next_step"
        ]
        
        for field in required_fields:
            if field not in data:
                return False
        
        return True
    
    # ============= KULLANICI EKLEME =============
    
    def kullanici_ekle(self, user_id, username, authority_level=AUTHORITY_LEVELS.VISITOR):
        """
        Yeni kullanıcı ekle
        
        Args:
            user_id (str): Kişi 1'den gelen user_id
            username (str): Kişi 1'den gelen recognized_user
            authority_level (int): Yetki seviyesi
        
        Returns:
            tuple: (success, result)
        """
        
        # Kontroller
        if self.active_users_count >= MAX_USERS:
            self._log("HATA", f"Bellek DOLU! Kullanıcı eklenemedi: {user_id}")
            return False, {"error": "Bellek kapasitesi dolu"}
        
        if user_id in self.user_index:
            self._log("UYARI", f"Kullanıcı zaten var: {user_id}")
            return False, {"error": "Bu ID zaten kayıtlı"}
        
        # Boş blok bul
        user_number, base_addr = self._find_empty_block()
        if base_addr is None:
            self._log("HATA", "Boş bellek bloğu bulunamadı")
            return False, {"error": "Boş blok bulunamadı"}
        
        # Belleğe yaz
        self.yazma(base_addr + MEMORY_OFFSET.USER_ID, user_id)
        self.yazma(base_addr + MEMORY_OFFSET.AUTHORITY_LEVEL, authority_level)
        self.yazma(base_addr + MEMORY_OFFSET.STATUS, USER_STATUS.ACTIVE)
        self.yazma(base_addr + MEMORY_OFFSET.FAILED_ATTEMPTS, 0)
        self.yazma(base_addr + MEMORY_OFFSET.ALARM, ALARM_STATUS.OFF)
        self.yazma(base_addr + MEMORY_OFFSET.LAST_ACCESS, None)
        
        # İndekse ekle
        self.user_index[user_id] = base_addr
        self.active_users_count += 1
        
        # Kullanıcı bilgilerini sakla
        self.users[user_id] = {
            "user_id": user_id,
            "name": username,
            "authority_level": authority_level,
            "status": USER_STATUS.ACTIVE,
            "failed_attempts": 0,
            "alarm": ALARM_STATUS.OFF,
            "last_access": None,
            "created_at": datetime.now().strftime(LOG_FORMAT),
            "base_address": hex(base_addr)
        }
        
        self._log("BASARILI", f"Kullanıcı eklendi: {user_id} ({username}, Adres: {hex(base_addr)})")
        
        # Durumu kaydet
        self._save_state()
        
        return True, {
            "user_id": user_id,
            "username": username,
            "base_address": hex(base_addr),
            "authority_level": authority_level,
            "message": "Kullanıcı başarıyla eklendi"
        }
    
    # ============= YAZMA İŞLEMLERİ =============
    
    def yazma(self, address, data):
        """Belleğe veri yaz (düşük seviye)"""
        self.memory[address] = data
        return True
    
    def okuma(self, address):
        """Bellekten veri oku (düşük seviye)"""
        return self.memory.get(address)
    
    # ============= OKUMA İŞLEMLERİ =============
    
    def kullanici_bilgisi_oku(self, user_id):
        """Kullanıcının tüm bilgilerini oku"""
        base_addr = self._find_user_address(user_id)
        
        if base_addr is None:
            return None
        
        return {
            "user_id": self.okuma(base_addr + MEMORY_OFFSET.USER_ID),
            "base_address": hex(base_addr),
            "authority_level": self.okuma(base_addr + MEMORY_OFFSET.AUTHORITY_LEVEL),
            "status": self.okuma(base_addr + MEMORY_OFFSET.STATUS),
            "failed_attempts": self.okuma(base_addr + MEMORY_OFFSET.FAILED_ATTEMPTS),
            "alarm": self.okuma(base_addr + MEMORY_OFFSET.ALARM),
            "last_access": self.okuma(base_addr + MEMORY_OFFSET.LAST_ACCESS)
        }
    
    def yetki_oku(self, user_id):
        """Yetki seviyesini oku"""
        base_addr = self._find_user_address(user_id)
        if base_addr is None:
            return None
        return self.okuma(base_addr + MEMORY_OFFSET.AUTHORITY_LEVEL)
    
    def durum_oku(self, user_id):
        """Durum bilgisini oku"""
        base_addr = self._find_user_address(user_id)
        if base_addr is None:
            return None
        return self.okuma(base_addr + MEMORY_OFFSET.STATUS)
    
    def alarm_oku(self, user_id):
        """Alarm durumunu oku"""
        base_addr = self._find_user_address(user_id)
        if base_addr is None:
            return None
        return self.okuma(base_addr + MEMORY_OFFSET.ALARM)
    
    def hata_sayisi_oku(self, user_id):
        """Hata sayısını oku"""
        base_addr = self._find_user_address(user_id)
        if base_addr is None:
            return None
        return self.okuma(base_addr + MEMORY_OFFSET.FAILED_ATTEMPTS)
    
    # ============= GÜNCELLEME İŞLEMLERİ =============
    
    def hata_sayisini_artir(self, user_id):
        """Hata sayısını 1 artır"""
        base_addr = self._find_user_address(user_id)
        if base_addr is None:
            return False
        
        current = self.okuma(base_addr + MEMORY_OFFSET.FAILED_ATTEMPTS)
        new_value = current + 1
        self.yazma(base_addr + MEMORY_OFFSET.FAILED_ATTEMPTS, new_value)
        
        self._log("GUNCELLEME", f"{user_id} hata sayısı: {current} -> {new_value}")
        
        # Kişi 3'e bildir
        if new_value >= 3:
            self.alarm_ac(user_id)
        
        return True
    
    def hata_sayisini_sifirla(self, user_id):
        """Hata sayısını sıfırla"""
        base_addr = self._find_user_address(user_id)
        if base_addr is None:
            return False
        
        self.yazma(base_addr + MEMORY_OFFSET.FAILED_ATTEMPTS, 0)
        self._log("GUNCELLEME", f"{user_id} hata sayısı sıfırlandı")
        
        return True
    
    def alarm_ac(self, user_id):
        """Alarm aç"""
        base_addr = self._find_user_address(user_id)
        if base_addr is None:
            return False
        
        self.yazma(base_addr + MEMORY_OFFSET.ALARM, ALARM_STATUS.ON)
        self._log("UYARI", f"{user_id} ALARM AÇILDI!")
        
        return True
    
    def alarm_kapat(self, user_id):
        """Alarm kapat"""
        base_addr = self._find_user_address(user_id)
        if base_addr is None:
            return False
        
        self.yazma(base_addr + MEMORY_OFFSET.ALARM, ALARM_STATUS.OFF)
        self._log("GUNCELLEME", f"{user_id} alarmı kapatıldı")
        
        return True
    
    def son_erisim_kaydet(self, user_id, zaman=None):
        """Son erişim zamanını kaydet"""
        base_addr = self._find_user_address(user_id)
        if base_addr is None:
            return False
        
        if zaman is None:
            zaman = datetime.now().strftime(LOG_FORMAT)
        
        self.yazma(base_addr + MEMORY_OFFSET.LAST_ACCESS, zaman)
        self._log("GUNCELLEME", f"{user_id} son erişim: {zaman}")
        
        return True
    
    def yetki_guncelle(self, user_id, new_authority):
        """Yetki seviyesini güncelle"""
        base_addr = self._find_user_address(user_id)
        if base_addr is None:
            return False
        
        old_authority = self.okuma(base_addr + MEMORY_OFFSET.AUTHORITY_LEVEL)
        self.yazma(base_addr + MEMORY_OFFSET.AUTHORITY_LEVEL, new_authority)
        
        self._log("GUNCELLEME", f"{user_id} yetki: {old_authority} -> {new_authority}")
        
        return True
    
    def durum_guncelle(self, user_id, new_status):
        """Durum bilgisini güncelle"""
        base_addr = self._find_user_address(user_id)
        if base_addr is None:
            return False
        
        status_name = "aktif" if new_status == USER_STATUS.ACTIVE else "pasif"
        self.yazma(base_addr + MEMORY_OFFSET.STATUS, new_status)
        
        self._log("GUNCELLEME", f"{user_id} durum: {status_name}")
        
        return True
    
    # ============= İSTATİSTİKLER =============
    
    def bellek_istatistikleri(self):
        """Bellek kullanım istatistikleri"""
        total = MAX_USERS
        used = self.active_users_count
        free = total - used
        percentage = (used / total * 100) if total > 0 else 0
        
        return {
            "total_capacity": total,
            "used": used,
            "free": free,
            "usage_percentage": round(percentage, 2),
            "address_range": f"0x{BASE_ADDRESS:02X} - 0x{BASE_ADDRESS + (MAX_USERS * ADDRESS_INCREMENT):02X}",
            "block_size": BLOCK_SIZE
        }
    
    def bellek_haritasi_goster(self):
        """Bellek haritasını ekrana göster"""
        print("\n" + "="*100)
        print("BELLEK HARİTASI")
        print("="*100)
        
        stats = self.bellek_istatistikleri()
        print(f"\nKapasite: {stats['used']}/{stats['total_capacity']} (%{stats['usage_percentage']})")
        print(f"Adres Aralığı: {stats['address_range']}\n")
        
        print(f"{'Adres':<10} {'User ID':<15} {'Ad':<15} {'Yetki':<12} {'Durum':<8} {'Hata':<6} {'Alarm':<10}")
        print("-"*100)
        
        if not self.user_index:
            print("(Bellek boş)")
        else:
            for user_id, base_addr in sorted(self.user_index.items()):
                addr_hex = hex(base_addr)
                authority = self.okuma(base_addr + MEMORY_OFFSET.AUTHORITY_LEVEL)
                status = self.okuma(base_addr + MEMORY_OFFSET.STATUS)
                failed = self.okuma(base_addr + MEMORY_OFFSET.FAILED_ATTEMPTS)
                alarm = self.okuma(base_addr + MEMORY_OFFSET.ALARM)
                
                authority_name = self._authority_name(authority)
                status_name = "Aktif" if status == USER_STATUS.ACTIVE else "Pasif"
                alarm_name = "AÇIK ⚠️" if alarm == ALARM_STATUS.ON else "Kapalı"
                
                username = self.users.get(user_id, {}).get("name", "?")
                
                print(f"{addr_hex:<10} {user_id:<15} {username:<15} {authority_name:<12} {status_name:<8} {failed:<6} {alarm_name:<10}")
        
        print("="*100 + "\n")
    
    def tum_kullanicilar(self):
        """Tüm kullanıcıları listele"""
        users_list = []
        for user_id in sorted(self.user_index.keys()):
            info = self.kullanici_bilgisi_oku(user_id)
            if info:
                users_list.append(info)
        return users_list
    
    # ============= LOG İŞLEMLERİ =============
    
    def _log(self, level, message):
        """Log kaydı yap"""
        timestamp = datetime.now().strftime(LOG_FORMAT)
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message
        }
        self.logs.append(log_entry)
        print(f"[{timestamp}] {level}: {message}")
    
    def log_goster(self):
        """Logları göster"""
        print("\n" + "="*100)
        print("ERIŞIM LOGLARI")
        print("="*100 + "\n")
        
        if not self.logs:
            print("(Log boş)")
        else:
            for entry in self.logs[-20:]:  # Son 20 logu göster
                print(f"[{entry['timestamp']}] {entry['level']}: {entry['message']}")
        
        print("\n" + "="*100 + "\n")
    
    # ============= YARDIMCI FONKSİYONLAR =============
    
    def _authority_name(self, level):
        """Yetki seviyesinin adını döndür"""
        names = {
            1: "Ziyaretçi",
            2: "Personel",
            3: "Yönetici"
        }
        return names.get(level, "Bilinmiyor")
    
    # ============= DURUM KAYDETME/YÜKLEME =============
    
    def _save_state(self):
        """Bellek durumunu dosyaya kaydet"""
        try:
            state = {
                "timestamp": datetime.now().strftime(LOG_FORMAT),
                "statistics": self.bellek_istatistikleri(),
                "users": self.users,
                "logs": self.logs[-100:]  # Son 100 logu kaydet
            }
            
            USERS_MEMORY_FILE.parent.mkdir(exist_ok=True)
            with open(USERS_MEMORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"❌ Durum kaydetme hatası: {e}")
            return False
    
    def _load_state(self):
        """Dosyadan bellek durumunu yükle"""
        try:
            if USERS_MEMORY_FILE.exists():
                with open(USERS_MEMORY_FILE, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                
                # Kullanıcıları yeniden yükle
                for user_id, user_info in state.get("users", {}).items():
                    authority = user_info.get("authority_level", AUTHORITY_LEVELS.VISITOR)
                    username = user_info.get("name", user_id)
                    self.kullanici_ekle(user_id, username, authority)
                
                self._log("SISTEM", f"{len(self.users)} kullanıcı yüklendi")
                return True
        except Exception as e:
            print(f"⚠️ Durum yükleme hatası: {e}")
            return False