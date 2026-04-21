# config.py
"""
Bellek Sistemi Ayarları
"""

import os
from pathlib import Path

# ============= DİZİN AYARLARI =============
CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR / "memory_data"
DATA_DIR.mkdir(exist_ok=True)

# ============= BELLEK AYARLARI =============

# Maksimum kapasite
MAX_USERS = 50
BLOCK_SIZE = 6  # Her kullanıcı 6 alan kaplar

# Adres ayarları
BASE_ADDRESS = 0x10
ADDRESS_INCREMENT = 0x10

# Bellek blok ofsetleri
class MEMORY_OFFSET:
    USER_ID = 0              # Kullanıcı ID
    AUTHORITY_LEVEL = 1      # Yetki seviyesi
    STATUS = 2               # Aktif/Pasif
    FAILED_ATTEMPTS = 3      # Başarısız giriş
    ALARM = 4                # Alarm durumu
    LAST_ACCESS = 5          # Son erişim zamanı

# ============= YETKİ SEVİYELERİ =============

class AUTHORITY_LEVELS:
    VISITOR = 1          # Ziyaretçi
    STAFF = 2            # Personel
    ADMIN = 3            # Yönetici

AUTHORITY_NAMES = {
    1: "Ziyaretçi",
    2: "Personel",
    3: "Yönetici"
}

# ============= DURUM DEĞERLERİ =============

class USER_STATUS:
    INACTIVE = 0
    ACTIVE = 1

STATUS_NAMES = {
    0: "Pasif",
    1: "Aktif"
}

# ============= ALARM DEĞERLERİ =============

class ALARM_STATUS:
    OFF = 0
    ON = 1

ALARM_NAMES = {
    0: "Kapalı",
    1: "AÇIK ⚠️"
}

# ============= KİŞİ 1'DEN BEKLENEN VERİ ALANLARI =============

KİŞİ1_EXPECTED_FIELDS = {
    "camera_status": str,
    "face_detection": str,
    "landmark_analysis": str,
    "pose": str,
    "recognized_user": str,
    "user_id": str,
    "confidence_score": float,
    "recognized": bool,
    "next_step": str
}

# ============= DOSYA YOLLARI =============

USERS_MEMORY_FILE = DATA_DIR / "users_memory.json"
MEMORY_DUMP_FILE = DATA_DIR / "memory_dump.json"
ADDRESS_TABLE_FILE = DATA_DIR / "address_table.json"
LOG_FILE = DATA_DIR / "access_log.json"

# ============= TANIMA EŞİKLERİ =============

# Kişi 1'den minimum güven skoru
MIN_CONFIDENCE_SCORE = 70.0

# Çoklu başarısız deneme
MAX_FAILED_ATTEMPTS = 3

# Tanındı = True olduğunda Kişi 2'nin yapacağı işlem
RECOGNIZED_CONFIDENCE_MIN = 75.0

# ============= LOG AYARLARI =============

LOG_ENABLED = True
LOG_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============= KİŞİ 3 İÇİN ÇIKTI AYARLARI =============

OUTPUT_TO_KISI3_ENABLED = True
OUTPUT_TO_KISI4_ENABLED = True

# ============= VERİ TABANINDA SAKLANACAK ALANLAR =============

STORED_USER_FIELDS = [
    "user_id",
    "name",
    "authority_level",
    "status",
    "failed_attempts",
    "alarm",
    "last_access",
    "created_at",
    "base_address"
]
# LBPH Yüz Tanıma Parametreleri (Kişi 1 için gerekli)
LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8