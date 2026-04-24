import os
from pathlib import Path

# --- TEMEL DİZİN YAPILANDIRMASI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_DIR = Path(__file__).parent

# --- KİŞİ 1 (YÜZ TANIMA) İHTİYAÇLARI ---
# face_utils.py ve train_model.py bu isimleri bekliyor
YUZZ_DIR = os.path.join(BASE_DIR, "yuzz")
USERS_JSON = os.path.join(BASE_DIR, "users.json")
TRAINER_PATH = os.path.join(BASE_DIR, "trainer.yml")

# Kamera Ayarları (Mobil Uyumlu - Daha Büyük)
CAMERA_INDEX = 0
FRAME_WIDTH = 640   # Daha büyük - yüzü görmek için
FRAME_HEIGHT = 480  # Daha büyük - yüzü görmek için
MIN_FACE_SIZE = 40  # Daha büyük minimum yüz boyutu

# LBPH Algoritma Parametreleri (Mobil Uyumlu - Dengeli)
LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8  # Daha hassas grid - daha iyi ayırt etme
LBPH_GRID_Y = 8  # Daha hassas grid - daha iyi ayırt etme
LBPH_STRICT_THRESHOLD = 85.0  # Daha makul strict eşik
LBPH_SOFT_THRESHOLD = 140.0   # Daha geniş soft eşik

# --- KİŞİ 2 (BELLEK SİSTEMİ) İHTİYAÇLARI ---
# bellek_utils.py bu isimleri ve sınıfları bekliyor
DATA_DIR = CURRENT_DIR / "memory_data"
DATA_DIR.mkdir(exist_ok=True)

MAX_USERS = 50
BLOCK_SIZE = 6
BASE_ADDRESS = 0x10
ADDRESS_INCREMENT = 0x10

USERS_MEMORY_FILE = DATA_DIR / "users_memory.json"
LOG_FORMAT = "%Y-%m-%d %H:%M:%S"

# --- KİŞİ 3 & 4 (KARAR VE ARAYÜZ) İHTİYAÇLARI ---
WINDOW_NAME_RECOGNIZE = "Yüz Tanıma Sistemi - Canlı Tanıma"
class MEMORY_OFFSET:
    USER_ID = 0
    AUTHORITY_LEVEL = 1
    STATUS = 2
    FAILED_ATTEMPTS = 3
    ALARM = 4
    LAST_ACCESS = 5

class AUTHORITY_LEVELS:
    VISITOR = 1
    STAFF = 2
    ADMIN = 3

class USER_STATUS:
    INACTIVE = 0
    ACTIVE = 1

class ALARM_STATUS:
    OFF = 0
    ON = 1

# Arayüzde görünen isimler (Kişi 4 için)
AUTHORITY_NAMES = {1: "Ziyaretçi", 2: "Personel", 3: "Yönetici"}
STATUS_NAMES = {0: "Pasif", 1: "Aktif"}
ALARM_NAMES = {0: "Kapalı", 1: "AÇIK"}

# Pencere İsimleri
WINDOW_NAME_RECOGNIZE = "Kisi1 - Tanima"
WINDOW_NAME_REGISTER = "Kisi1 - Yeni Kullanici Kaydi"
MIN_CONFIDENCE_SCORE = 60.0  # Güven skoru %60 altındaysa reddet (daha gerçekçi)
LOG_ENABLED = True           # İşlemleri kaydet
# --- BU SATIRLAR SENİN KODUNDA EKSİK OLAN VE HATAYA SEBEP OLAN KISIMLAR ---
MEMORY_DUMP_FILE = DATA_DIR / "memory_dump.txt"
# Görüntü netlik sınırı (Bulanık fotoğrafları reddetmek için)
BLUR_THRESHOLD = 20  # Bulanık görüntüleri reddet (biraz daha toleranslı)
BRIGHTNESS_THRESHOLD = 30  # Karanlık görüntüleri reddet
MIN_DISPLAY_SCORE_TO_ACCEPT = 40  # Minimum kabul skoru
MIN_REPEATS_FRONT = 4   # Ön yüz için minimum tekrar
MIN_REPEATS_PROFILE = 4  # Profil için minimum tekrar
MIN_REPEATS_UPDOWN = 4   # Yukarı/aşağı için minimum tekrar
# --- DOMINANCE FARK (Kararlılık) ---
DOMINANCE_DIFF = 1  # En iyi 2 tahmin arasında minimum fark

# --- CANLI TANIMA PARAMETRELERI ---
HISTORY_WINDOW = 5   # Daha kısa pencere - daha hızlı karar
PREDICT_EVERY_N_FRAMES = 2  # Daha sık tahmin

# --- TEST MODU AYARLARI ---
TEST_MODE = True

# Test modunda kullanılan daha toleranslı eşikler
# Tanınmayan kişiler için yanlış isim önlemek: eşikleri daha makul ayarla
TEST_LBPH_SOFT_THRESHOLD = 70.0
TEST_MIN_DISPLAY_SCORE_TO_ACCEPT = 65.0
TEST_BLUR_THRESHOLD = 5.0
TEST_BRIGHTNESS_THRESHOLD = 10.0
TEST_DOMINANCE_DIFF = 4
