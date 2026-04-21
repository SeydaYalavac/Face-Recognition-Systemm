# ============================================================================
# YÜZ TANIMA SİSTEMİ - BÜTÜNLEŞTIRILMIŞ VERSIYON
# ============================================================================
# Tüm fonksiyonlar ve sınıflar bu dosyada birleştirilmiştir
# ============================================================================

import os
import json
import cv2
import numpy as np
import datetime
import shutil
import glob
from datetime import datetime as dt
from pathlib import Path

# MediaPipe kütüphaneleri
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

# Plyer kütüphaneleri (donanım tetikleme için)
try:
    from plyer import vibrator, tts, notification
    PLYER_HAZIR = True
except ImportError:
    PLYER_HAZIR = False


# ============================================================================
# BÖLÜM 1: KONFİGURASYON AYARLARI (config.py)
# ============================================================================

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

# Ayarlar
CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR / "memory_data"
DATA_DIR.mkdir(exist_ok=True)

# Bellek ayarları
MAX_USERS = 50
BLOCK_SIZE = 6
BASE_ADDRESS = 0x10
ADDRESS_INCREMENT = 0x10

# Dosya yolları
USERS_MEMORY_FILE = DATA_DIR / "users_memory.json"
MEMORY_DUMP_FILE = DATA_DIR / "memory_dump.json"
ADDRESS_TABLE_FILE = DATA_DIR / "address_table.json"
LOG_FILE = DATA_DIR / "access_log.json"
USERS_JSON = DATA_DIR / "users.json"
YUZZ_DIR = DATA_DIR / "yuzler"
TRAINER_PATH = DATA_DIR / "face_recognizer.yml"

# Eşikler
MIN_CONFIDENCE_SCORE = 70.0
MAX_FAILED_ATTEMPTS = 3
RECOGNIZED_CONFIDENCE_MIN = 75.0
GUVEN_ESIGI = 75.0

# Log ayarları
LOG_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_ENABLED = True
LOG_DOSYASI = "erisim_log.json"

# LBPH parametreleri
LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8

# Beklenen veri alanları
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

AUTHORITY_NAMES = {1: "Ziyaretçi", 2: "Personel", 3: "Yönetici"}
STATUS_NAMES = {0: "Pasif", 1: "Aktif"}
ALARM_NAMES = {0: "Kapalı", 1: "AÇIK ⚠️"}


# ============================================================================
# BÖLÜM 2: YÜZ İŞLEME ARAÇLARI (face_utils.py)
# ============================================================================

# MediaPipe modelleri
if MP_AVAILABLE:
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )
    
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


def ensure_paths():
    """Gerekli klasörleri ve dosyaları oluştur"""
    os.makedirs(YUZZ_DIR, exist_ok=True)
    if not os.path.exists(USERS_JSON):
        with open(USERS_JSON, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)


def load_users():
    """Kullanıcı listesini yükle"""
    ensure_paths()
    with open(USERS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(users):
    """Kullanıcı listesini kaydet"""
    ensure_paths()
    with open(USERS_JSON, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def find_user_by_numeric_id(numeric_id: int):
    """Sayısal ID'ye göre kullanıcı bul"""
    users = load_users()
    for user in users:
        if user["numeric_id"] == int(numeric_id):
            return user
    return None


def detect_face_and_crop(frame):
    """Kameradan alınan framede yüzü tespit et ve kırp"""
    if not MP_AVAILABLE:
        return None, None
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if not results.detections:
        return None, None

    h, w, _ = frame.shape
    best_face = None
    best_area = 0

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        x = max(int(bbox.xmin * w), 0)
        y = max(int(bbox.ymin * h), 0)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        pad_x = int(bw * 0.12)
        pad_y = int(bh * 0.12)

        x1 = max(x - pad_x, 0)
        y1 = max(y - pad_y, 0)
        x2 = min(x + bw + pad_x, w)
        y2 = min(y + bh + pad_y, h)

        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best_face = (x1, y1, x2, y2)

    if best_face is None:
        return None, None

    x1, y1, x2, y2 = best_face
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None, None

    return best_face, crop


def preprocess_face(face_bgr):
    """Yüz görüntüsünü ön işleme tabi tut"""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (200, 200))

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    mean_val = float(gray.mean())
    if mean_val < 90:
        gray = cv2.convertScaleAbs(gray, alpha=1.4, beta=20)
    elif mean_val < 110:
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def load_and_crop_image_for_training(img_path):
    """Eğitim için resim yükle ve kırp"""
    img = cv2.imread(img_path)
    if img is None:
        return None

    _, face_crop = detect_face_and_crop(img)

    if face_crop is None:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (200, 200))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            return gray
        except Exception:
            return None

    return preprocess_face(face_crop)


def estimate_pose_label(frame):
    """Yüz pozisyonunu tahmin et (on, sol, sağ, yukarı, aşağı)"""
    if not MP_AVAILABLE:
        return "unknown", {"yaw": 0.0, "pitch": 0.0}
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return "unknown", {"yaw": 0.0, "pitch": 0.0}

    h, w, _ = frame.shape
    landmarks = results.multi_face_landmarks[0].landmark

    nose = landmarks[1]
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]
    forehead = landmarks[10]
    chin = landmarks[152]

    nose_x = nose.x * w
    nose_y = nose.y * h
    left_x = left_cheek.x * w
    right_x = right_cheek.x * w
    forehead_y = forehead.y * h
    chin_y = chin.y * h

    face_center_x = (left_x + right_x) / 2.0
    face_width = max(abs(right_x - left_x), 1.0)
    face_center_y = (forehead_y + chin_y) / 2.0
    face_height = max(abs(chin_y - forehead_y), 1.0)

    yaw_ratio = (nose_x - face_center_x) / face_width
    pitch_ratio = (nose_y - face_center_y) / face_height

    yaw_deg = yaw_ratio * 100
    pitch_deg = pitch_ratio * 100

    if yaw_deg < -12:
        pose = "right"
    elif yaw_deg > 12:
        pose = "left"
    elif pitch_deg < -10:
        pose = "up"
    elif pitch_deg > 10:
        pose = "down"
    else:
        pose = "front"

    return pose, {"yaw": yaw_deg, "pitch": pitch_deg}


def measure_blur(face_bgr):
    """Yüz bulanıklığını ölç"""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def measure_brightness(face_bgr):
    """Yüz parlaklığını ölç"""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def draw_status_panel(frame, panel_data):
    """Kamera durumu paneli çiz"""
    cv2.rectangle(frame, (10, 10), (640, 305), (40, 40, 40), -1)
    cv2.rectangle(frame, (10, 10), (640, 305), (0, 255, 255), 2)

    lines = [
        f"Kamera durumu: {panel_data.get('camera_status', '-')}",
        f"Yüz algılama: {panel_data.get('face_detection', '-')}",
        f"Landmark analiz: {panel_data.get('landmark_analysis', '-')}",
        f"Pozisyon: {panel_data.get('pose', '-')}",
        f"Tanınan kullanıcı: {panel_data.get('recognized_user', '-')}",
        f"Güven skoru: {panel_data.get('confidence_score', 0):.2f}%",
        f"Durum: {'TANINDI' if panel_data.get('recognized') else 'TANINMADI'}",
    ]

    for i, line in enumerate(lines):
        y_pos = 40 + i * 35
        cv2.putText(frame, line, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return frame


# ============================================================================
# BÖLÜM 3: ÇIKTI FORMATLAMA (output_formatter.py)
# ============================================================================

def build_output(user_id, name, recognized, confidence, pose, next_step):
    """Kişi 1'in çıktısını formatla"""
    return {
        "timestamp": dt.now().strftime("%Y-%m-%d %H:%M:%S"),
        "camera_status": "aktif",
        "face_detection": "basarili",
        "landmark_analysis": "tamamlandi",
        "pose": pose,
        "recognized_user": name,
        "user_id": user_id,
        "confidence_score": round(confidence, 2),
        "recognized": recognized,
        "next_step": next_step
    }


# ============================================================================
# BÖLÜM 4: BELLEK SİSTEMİ (bellek_utils.py)
# ============================================================================

class BellekSistemi:
    """Bellek Sistemi Yöneticisi - Kişi 2"""
    
    def __init__(self):
        """Bellek sistemini başlat"""
        self.memory = {}
        self.user_index = {}
        self.users = {}
        self.logs = []
        self.active_users_count = 0
        
        self._log("SISTEM", "Bellek Sistemi başlatıldı")
        self._log("SİSTEM", f"Max kapasite: {MAX_USERS} kullanıcı, Blok boyutu: {BLOCK_SIZE} alan")
        self._load_state()
    
    # ===== ADRES İŞLEMLERİ =====
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
    
    # ===== KİŞİ 1'DEN VERİ İŞLEME =====
    def kisi1_verisini_isle(self, kisi1_data):
        """Kişi 1'den gelen veriyi işle"""
        if not self._validate_kisi1_data(kisi1_data):
            return {"success": False, "error": "Kişi 1 verisi geçersiz", "data": kisi1_data}
        
        if kisi1_data["camera_status"] != "aktif":
            self._log("UYARI", "Kamera aktif değil")
            return {"success": False, "error": "Kamera kapalı", "access_granted": False}
        
        if kisi1_data["face_detection"] != "basarili":
            self._log("UYARI", "Yüz algılanamadı")
            return {"success": False, "error": "Yüz algılanamadı", "access_granted": False}
        
        if not kisi1_data["recognized"]:
            self._log("UYARI", f"Tanıma başarısız: {kisi1_data['next_step']}")
            return {"success": False, "error": kisi1_data["next_step"], "access_granted": False}
        
        if kisi1_data["confidence_score"] < MIN_CONFIDENCE_SCORE:
            self._log("UYARI", f"Güven skoru düşük: {kisi1_data['confidence_score']}")
            return {"success": False, "error": f"Güven skoru yetersiz", "access_granted": False}
        
        user_id = kisi1_data["user_id"]
        username = kisi1_data["recognized_user"]
        confidence = kisi1_data["confidence_score"]
        
        self._log("BASARILI", f"Tanıma başarılı: {username} ({user_id}, Güven: {confidence}%)")
        
        user_info = self.kullanici_bilgisi_oku(user_id)
        
        if user_info is None:
            success, result = self.kullanici_ekle(user_id, username)
            if not success:
                return {"success": False, "error": "Kullanıcı belleğe eklenemedi", "access_granted": False}
        
        self.son_erisim_kaydet(user_id)
        self.hata_sayisini_sifirla(user_id)
        self.alarm_kapat(user_id)
        
        return {
            "success": True,
            "user_id": user_id,
            "username": username,
            "confidence": confidence,
            "access_granted": True,
            "kisi1_data": kisi1_data
        }
    
    def _validate_kisi1_data(self, data):
        """Kişi 1 verisini doğrula"""
        required_fields = ["camera_status", "face_detection", "landmark_analysis",
                          "pose", "recognized_user", "user_id", "confidence_score",
                          "recognized", "next_step"]
        return all(field in data for field in required_fields)
    
    # ===== KULLANICI EKLEME =====
    def kullanici_ekle(self, user_id, username, authority_level=AUTHORITY_LEVELS.VISITOR):
        """Yeni kullanıcı ekle"""
        if self.active_users_count >= MAX_USERS:
            self._log("HATA", f"Bellek DOLU! Kullanıcı eklenemedi: {user_id}")
            return False, {"error": "Bellek kapasitesi dolu"}
        
        if user_id in self.user_index:
            self._log("UYARI", f"Kullanıcı zaten var: {user_id}")
            return False, {"error": "Bu ID zaten kayıtlı"}
        
        user_number, base_addr = self._find_empty_block()
        if base_addr is None:
            self._log("HATA", "Boş bellek bloğu bulunamadı")
            return False, {"error": "Boş blok bulunamadı"}
        
        self.yazma(base_addr + MEMORY_OFFSET.USER_ID, user_id)
        self.yazma(base_addr + MEMORY_OFFSET.AUTHORITY_LEVEL, authority_level)
        self.yazma(base_addr + MEMORY_OFFSET.STATUS, USER_STATUS.ACTIVE)
        self.yazma(base_addr + MEMORY_OFFSET.FAILED_ATTEMPTS, 0)
        self.yazma(base_addr + MEMORY_OFFSET.ALARM, ALARM_STATUS.OFF)
        self.yazma(base_addr + MEMORY_OFFSET.LAST_ACCESS, None)
        
        self.user_index[user_id] = base_addr
        self.active_users_count += 1
        
        self.users[user_id] = {
            "user_id": user_id,
            "name": username,
            "authority_level": authority_level,
            "status": USER_STATUS.ACTIVE,
            "failed_attempts": 0,
            "alarm": ALARM_STATUS.OFF,
            "last_access": None,
            "created_at": dt.now().strftime(LOG_FORMAT),
            "base_address": hex(base_addr)
        }
        
        self._log("BASARILI", f"Kullanıcı eklendi: {user_id} ({username}, Adres: {hex(base_addr)})")
        self._save_state()
        
        return True, {
            "user_id": user_id,
            "username": username,
            "base_address": hex(base_addr),
            "authority_level": authority_level,
            "message": "Kullanıcı başarıyla eklendi"
        }
    
    # ===== YAZMA/OKUMA İŞLEMLERİ =====
    def yazma(self, address, data):
        """Belleğe veri yaz"""
        self.memory[address] = data
        return True
    
    def okuma(self, address):
        """Bellekten veri oku"""
        return self.memory.get(address)
    
    # ===== OKUMA İŞLEMLERİ =====
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
    
    # ===== GÜNCELLEME İŞLEMLERİ =====
    def hata_sayisini_artir(self, user_id):
        """Hata sayısını 1 artır"""
        base_addr = self._find_user_address(user_id)
        if base_addr is None:
            return False
        
        current = self.okuma(base_addr + MEMORY_OFFSET.FAILED_ATTEMPTS)
        new_value = current + 1
        self.yazma(base_addr + MEMORY_OFFSET.FAILED_ATTEMPTS, new_value)
        
        self._log("GUNCELLEME", f"{user_id} hata sayısı: {current} -> {new_value}")
        
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
            zaman = dt.now().strftime(LOG_FORMAT)
        
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
    
    # ===== İSTATİSTİKLER =====
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
    
    # ===== LOG İŞLEMLERİ =====
    def _log(self, level, message):
        """Log kaydı yap"""
        timestamp = dt.now().strftime(LOG_FORMAT)
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
            for entry in self.logs[-20:]:
                print(f"[{entry['timestamp']}] {entry['level']}: {entry['message']}")
        
        print("\n" + "="*100 + "\n")
    
    # ===== YARDIMCI FONKSİYONLAR =====
    def _authority_name(self, level):
        """Yetki seviyesinin adını döndür"""
        return AUTHORITY_NAMES.get(level, "Bilinmiyor")
    
    # ===== DURUM KAYIT/YÜKLEMESİ =====
    def _save_state(self):
        """Sistemin durumunu kaydet"""
        try:
            state = {
                "users": self.users,
                "active_users_count": self.active_users_count,
                "user_index": {k: hex(v) for k, v in self.user_index.items()}
            }
            with open(USERS_MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log("HATA", f"Durum kaydedilemedi: {e}")
    
    def _load_state(self):
        """Önceki durumu yükle"""
        try:
            if os.path.exists(USERS_MEMORY_FILE):
                with open(USERS_MEMORY_FILE, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    self.users = state.get("users", {})
                    self.active_users_count = state.get("active_users_count", 0)
                    
                    for user_id, hex_addr in state.get("user_index", {}).items():
                        self.user_index[user_id] = int(hex_addr, 16)
                        base_addr = self.user_index[user_id]
                        
                        for offset in range(BLOCK_SIZE):
                            key = base_addr + offset
                            self.memory[key] = None
        except Exception as e:
            self._log("HATA", f"Durum yüklenemedi: {e}")


# ============================================================================
# BÖLÜM 5: ERİŞİM KONTROL (access_control.py)
# ============================================================================

def donanim_tetikle(durum, mesaj):
    """Donanım tetiklemesi (titreşim, ses, bildirim)"""
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
    """Erişim logunu kaydet"""
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
    """Erişim kararı üret (Kişi 3 - Kişi 2 entegrasyonu)"""
    bellek_verisi = bellek_interface.get_user_data(user_id)
    
    sonuc = {
        "access_granted": False,
        "message": "Erisim Reddedildi",
        "alarm": 0,
        "timestamp": dt.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if not bellek_verisi.get("found", False):
        sonuc["message"] = "HATA: Kullanici kaydi bulunamadi!"
        donanim_tetikle("red", sonuc["message"])
        return sonuc

    if not yuz_verisi.get("recognized", False) or yuz_verisi.get("confidence_score", 0) < GUVEN_ESIGI:
        bellek_interface.record_failure(user_id)
        
        yeni_hata_sayisi = bellek_verisi.get('failed_attempts', 0) + 1
        sonuc["message"] = f"DUSUK GUVEN VEYA HATALI YUZ! (Hata: {yeni_hata_sayisi}/{MAX_HATALI_GIRIS})"
        donanim_tetikle("red", sonuc["message"])
        
        if yeni_hata_sayisi >= MAX_HATALI_GIRIS:
            sonuc["alarm"] = 1
            sonuc["message"] = "!!! ALARM AKTIF: COKLU HATALI GIRIS !!!"
            donanim_tetikle("alarm", sonuc["message"])
            
    elif bellek_verisi.get("status") == 1 and bellek_verisi.get("authority_level", 0) >= 1:
        bellek_interface.record_success(user_id)
        sonuc["access_granted"] = True
        sonuc["message"] = f"Erisim Onaylandi. Hos geldiniz, {user_id}."
        donanim_tetikle("onay", "Giris Basarili")
    
    else:
        sonuc["message"] = "Yetkisiz veya Pasif Kullanici Durumu."
        donanim_tetikle("red", sonuc["message"])

    log_kaydet({**yuz_verisi, **sonuc, "auth_level": bellek_verisi.get("authority_level")})
    return sonuc


def log_goruntule():
    """Kayıtlı logları göster"""
    if os.path.exists(LOG_DOSYASI):
        with open(LOG_DOSYASI, "r", encoding="utf-8") as f:
            try:
                loglar = json.load(f)
                print("\n" + "="*55)
                print("       --- SİSTEM ERİŞİM KAYITLARI (LOGLAR) ---")
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


# ============================================================================
# BÖLÜM 6: MODELİ EĞITME (train_model.py)
# ============================================================================

def train_lbph_model():
    """LBPH (Local Binary Patterns Histograms) modelini eğit"""
    users = load_users()
    if not users:
        print("Egitilecek kullanici yok. users.json dosyasini kontrol edin.")
        return

    faces = []
    labels = []

    for user in users:
        numeric_id = user["numeric_id"]
        folder_path = os.path.join(YUZZ_DIR, user["folder"])

        if not os.path.exists(folder_path):
            print(f"Klasor yok: {folder_path}")
            continue

        added_count = 0
        skipped_count = 0

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(folder_path, file_name)
                processed = load_and_crop_image_for_training(img_path)

                if processed is None:
                    skipped_count += 1
                    continue

                faces.append(processed)
                labels.append(numeric_id)
                added_count += 1

        print(f"{user['name']} icin egitime eklenen: {added_count}, atlanan: {skipped_count}")

    if not faces:
        print("Egitim icin gecerli goruntu bulunamadi.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=LBPH_RADIUS,
        neighbors=LBPH_NEIGHBORS,
        grid_x=LBPH_GRID_X,
        grid_y=LBPH_GRID_Y
    )

    recognizer.train(faces, np.array(labels))
    recognizer.write(TRAINER_PATH)

    print(f"Model egitildi: {TRAINER_PATH}")
    print(f"Toplam goruntu: {len(faces)}")


# ============================================================================
# BÖLÜM 7: KULLANICI FOTOĞRAFLARINı TEMİZLEME
# ============================================================================

def clear_user_folder(folder_name="yuz3"):
    """Belirtilen klasördeki tüm fotoğrafları sil"""
    folder_path = os.path.join(YUZZ_DIR, folder_name)

    if not os.path.exists(folder_path):
        print(f"Klasör bulunamadı: {folder_path}")
        return False

    file_count = 0
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                file_count += 1
        except Exception as e:
            print(f"Hata: {file} silinemedi → {e}")

    print(f"{folder_name} klasöründeki {file_count} fotoğraf silindi.")
    print(f"Şimdi '1 - Yeni kullanici kaydet' ile tekrar {folder_name} klasörüne kayıt alabilirsiniz.")
    return True


def clear_temp_user_photos():
    """Geçici kullanıcıların (numeric_id >= 3) fotoğraflarını sil"""
    users = load_users()

    if not users:
        print("Kayitli kullanici bulunamadi.")
        return

    silinen_toplam = 0
    islenen_kullanici = 0

    for user in users:
        if user["numeric_id"] <= 2:
            print(f"Korundu: {user['name']} ({user['user_id']})")
            continue

        folder_path = os.path.join(YUZZ_DIR, user["folder"])

        if not os.path.exists(folder_path):
            print(f"Klasor bulunamadi: {folder_path}")
            continue

        files = (
            glob.glob(os.path.join(folder_path, "*.jpg")) +
            glob.glob(os.path.join(folder_path, "*.jpeg")) +
            glob.glob(os.path.join(folder_path, "*.png"))
        )

        if not files:
            print(f"{user['name']} icin silinecek fotograf yok.")
            continue

        for f in files:
            os.remove(f)

        silinen_toplam += len(files)
        islenen_kullanici += 1

        print(f"Silindi: {user['name']} ({user['user_id']}) - {len(files)} fotograf")

    print(f"\nToplam {islenen_kullanici} kullanicinin {silinen_toplam} fotografu silindi.")
    print("Seyda ve Sevde korundu.")


def clear_single_user_photos(user_name: str):
    """Belirtilen kullanıcının fotoğraflarını sil"""
    users = load_users()
    target_user = None

    for user in users:
        if user["name"].strip().lower() == user_name.strip().lower():
            target_user = user
            break

    if target_user is None:
        print(f"Kullanici bulunamadi: {user_name}")
        return

    if target_user["numeric_id"] <= 2:
        print(f"UYARI: {target_user['name']} korunan kullanicidir. Fotograflari silinemez.")
        return

    folder_path = os.path.join(YUZZ_DIR, target_user["folder"])

    if not os.path.exists(folder_path):
        print(f"Klasor bulunamadi: {folder_path}")
        return

    files = (
        glob.glob(os.path.join(folder_path, "*.jpg")) +
        glob.glob(os.path.join(folder_path, "*.jpeg")) +
        glob.glob(os.path.join(folder_path, "*.png"))
    )

    if not files:
        print(f"{user_name} icin silinecek fotograf bulunamadi.")
        return

    for f in files:
        os.remove(f)

    print(f"{user_name} icin {len(files)} fotograf silindi.")
    print(f"Klasor korundu: {folder_path}")


# ============================================================================
# BÖLÜM 8: KİŞİ 4 ARAYÜZÜ (kisi4_interface.py)
# ============================================================================

class Kisi4Interface:
    """Kişi 4'ün kullanacağı interface - Ham veri sağlama"""
    
    def __init__(self, bellek):
        self.bellek = bellek
    
    def get_memory_map(self):
        """Bellek haritası verilerini ver"""
        users = []
        for user in self.bellek.tum_kullanicilar():
            users.append({
                "base_address": user["base_address"],
                "user_id": user["user_id"],
                "authority_level": user["authority_level"],
                "status": user["status"],
                "failed_attempts": user["failed_attempts"],
                "alarm": user["alarm"],
                "last_access": user["last_access"]
            })
        
        return {
            "statistics": self.bellek.bellek_istatistikleri(),
            "users": users
        }
    
    def get_dashboard(self):
        """Dashboard için ham veri"""
        stats = self.bellek.bellek_istatistikleri()
        users = self.bellek.tum_kullanicilar()
        
        return {
            "memory_stats": stats,
            "user_count": len(users),
            "active_count": len([u for u in users if u["status"] == 1]),
            "alarm_count": len([u for u in users if u["alarm"] == 1])
        }
    
    def get_user_data(self, user_id):
        """Kullanıcı verilerini getir"""
        info = self.bellek.kullanici_bilgisi_oku(user_id)
        if info is None:
            return {"found": False}
        
        return {
            "found": True,
            "user_id": info["user_id"],
            "authority_level": info["authority_level"],
            "status": info["status"],
            "failed_attempts": info["failed_attempts"],
            "alarm": info["alarm"],
            "last_access": info["last_access"]
        }
    
    def record_success(self, user_id):
        """Başarılı girişi kaydet"""
        self.bellek.hata_sayisini_sifirla(user_id)
        self.bellek.alarm_kapat(user_id)
        self.bellek.son_erisim_kaydet(user_id)
    
    def record_failure(self, user_id):
        """Başarısız girişi kaydet"""
        self.bellek.hata_sayisini_artir(user_id)


# ============================================================================
# BÖLÜM 9: ANA PROGRAM (main)
# ============================================================================

def ana_menu():
    """Ana menüyü göster ve seçim al"""
    print("\n" + "="*60)
    print("     YÜZ TANIMA SİSTEMİ - BÜTÜNLEŞTIRILMIŞ VERSIYON")
    print("="*60)
    print("\n1 - Yeni kullanici kaydet")
    print("2 - Modeli egit")
    print("3 - Canli tanima baslat")
    print("4 - Yeni kullanicilarin fotograflarini sil (yuz3+)")
    print("5 - Tek kullanicinin fotograflarini sil")
    print("6 - Bellek haritasini goster")
    print("7 - Erisim kayitlarini goster")
    print("8 - Cikis")
    print("="*60)


def main():
    """Ana program - Tüm modülleri koordine et"""
    
    # Başlatma
    ensure_paths()
    bellek = BellekSistemi()
    interface = Kisi4Interface(bellek)
    
    print("\n✓ Sistem başlatıldı!")
    print(f"✓ Bellek kapasite: {bellek.bellek_istatistikleri()['total_capacity']} kullanıcı")
    print(f"✓ Aktif kullanıcılar: {bellek.bellek_istatistikleri()['used']}")
    
    while True:
        ana_menu()
        secim = input("Seciminiz: ").strip()

        if secim == "1":
            name = input("Kullanici adi: ").strip()
            if name:
                print(f"Yeni kullanici kaydi başlatılıyor: {name}")
                # Burada kayıt fonksiyonu çağrılacak
                print(f"{name} için fotoğraf toplama başladı...")
            else:
                print("Gecersiz isim.")

        elif secim == "2":
            print("\nModel eğitimi başlatılıyor...")
            train_lbph_model()

        elif secim == "3":
            print("\nCanlı tanıma başlatılıyor...")
            print("(Kamera açılacak - Q tuşu ile çıkın)")
            # recognize_live() çağrısı burada yapılacak

        elif secim == "4":
            print("\nSeyda ve Sevde korunacak.")
            print("Diger tum kullanicilarin fotograflari silinecek.")
            onay = input("Emin misiniz? (evet/hayir): ").strip().lower()
            if onay == "evet":
                clear_temp_user_photos()
            else:
                print("Iptal edildi.")

        elif secim == "5":
            name = input("Fotograflari silinecek kullanici adi: ").strip()
            if name:
                clear_single_user_photos(name)
            else:
                print("Gecersiz isim.")

        elif secim == "6":
            bellek.bellek_haritasi_goster()

        elif secim == "7":
            bellek.log_goster()

        elif secim == "8":
            print("\nCikis yapildi. Hoşça kalın!")
            break

        else:
            print("Gecersiz secim. Lütfen tekrar deneyin.")


if __name__ == "__main__":
    main()
