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

# Kamera Ayarları
CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 720
MIN_FACE_SIZE = 100

# LBPH Algoritma Parametreleri
LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8
LBPH_STRICT_THRESHOLD = 55.0
LBPH_SOFT_THRESHOLD = 70.0

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
MIN_CONFIDENCE_SCORE = 70.0  # Güven skoru %70 altındaysa reddet
LOG_ENABLED = True           # İşlemleri kaydet
# --- BU SATIRLAR SENİN KODUNDA EKSİK OLAN VE HATAYA SEBEP OLAN KISIMLAR ---
MEMORY_DUMP_FILE = DATA_DIR / "memory_dump.txt"
# Görüntü netlik sınırı (Bulanık fotoğrafları reddetmek için)
BLUR_THRESHOLD =40