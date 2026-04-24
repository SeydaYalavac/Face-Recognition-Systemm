import os
import json
import cv2
import numpy as np
import mediapipe as mp
import psutil

from config import USERS_JSON, YUZZ_DIR
BLUR_THRESHOLD = 100.0
# --- MediaPipe Tasks API (Yeni Versiyon) ---
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Model dosyaları
FACE_DETECTOR_MODEL = os.path.join(os.path.dirname(__file__), 'blaze_face_short_range.tflite')
FACE_LANDMARKER_MODEL = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')

# Mobil uyumlu face detector (video için)
face_detector_options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=FACE_DETECTOR_MODEL),
    running_mode=VisionRunningMode.VIDEO,
    min_detection_confidence=0.2,  # Daha toleranslı
    min_suppression_threshold=0.2
)

# Eğitim için image mode face detector
face_detector_image_options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=FACE_DETECTOR_MODEL),
    running_mode=VisionRunningMode.IMAGE,
    min_detection_confidence=0.2,  # Daha toleranslı
    min_suppression_threshold=0.2
)

# Face landmarker for pose estimation
face_landmarker_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL),
    running_mode=VisionRunningMode.VIDEO,
    min_face_detection_confidence=0.3,
    min_face_presence_confidence=0.3,
    min_tracking_confidence=0.3,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

face_detector = None
face_detector_image = None  # Eğitim için
face_landmarker = None

# MediaPipe video mode için monotonik artan timestamp
_mp_timestamp = 0


def _get_next_timestamp():
    """MediaPipe video mode için her çağrıda artan timestamp üretir."""
    global _mp_timestamp
    _mp_timestamp += 33  # ~30fps
    return _mp_timestamp

def init_face_detector():
    """MediaPipe face detector ve landmarker'ı başlatır."""
    global face_detector, face_detector_image, face_landmarker

    # Model dosyalarının varlığını kontrol et
    if not os.path.exists(FACE_DETECTOR_MODEL):
        print(f"❌ Face detector model dosyası bulunamadı: {FACE_DETECTOR_MODEL}")
        return False

    if not os.path.exists(FACE_LANDMARKER_MODEL):
        print(f"❌ Face landmarker model dosyası bulunamadı: {FACE_LANDMARKER_MODEL}")
        return False

    if face_detector is None:
        try:
            face_detector = FaceDetector.create_from_options(face_detector_options)
            print("✅ MediaPipe Face Detector (Video) başlatıldı")
        except Exception as e:
            print(f"❌ MediaPipe Face Detector (Video) başlatılamadı: {e}")
            face_detector = None
            return False

    if face_detector_image is None:
        try:
            face_detector_image = FaceDetector.create_from_options(face_detector_image_options)
            print("✅ MediaPipe Face Detector (Image) başlatıldı")
        except Exception as e:
            print(f"❌ MediaPipe Face Detector (Image) başlatılamadı: {e}")
            face_detector_image = None
            return False

    if face_landmarker is None:
        try:
            face_landmarker = FaceLandmarker.create_from_options(face_landmarker_options)
            print("✅ MediaPipe Face Landmarker başlatıldı")
        except Exception as e:
            print(f"❌ MediaPipe Face Landmarker başlatılamadı: {e}")
            face_landmarker = None
            return False

    return True

# Başlatma
init_success = init_face_detector()


# --- Sistem Performans Fonksiyonları ---
def get_system_performance():
    """Sistem performans verilerini toplar."""
    try:
        return {
            "cpu_usage": psutil.cpu_percent(interval=None),
            "ram_usage": psutil.virtual_memory().percent,
            "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else 0
        }
    except Exception as e:
        return {"cpu_usage": 0, "ram_usage": 0, "frequency": 0}


def check_device_stability(sensor_data, threshold=1.2):
    """
    Sarsıntı Analizi - Hata vermeyen (Safe) versiyon.
    """
    try:
        if sensor_data is None:
            return True

        if len(sensor_data) < 3:
            return True

        x, y, z = sensor_data[0], sensor_data[1], sensor_data[2]
        magnitude = (x ** 2 + y ** 2 + z ** 2) ** 0.5

        return magnitude < threshold
    except Exception as e:
        print(f"Sensör Hatası: {e}")
        return True


# --- Dizin ve JSON Yönetimi ---
def ensure_paths():
    """Gerekli dizinleri ve dosyaları oluşturur."""
    os.makedirs(YUZZ_DIR, exist_ok=True)
    if not os.path.exists(USERS_JSON):
        with open(USERS_JSON, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)


def load_users():
    """Kullanıcı listesini JSON'dan yükler."""
    ensure_paths()
    with open(USERS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def find_user_by_numeric_id(numeric_id):
    """Numeric ID'ye göre kullanıcı bulur."""
    users_data = load_users()

    # Liste formatında kontrol
    if isinstance(users_data, list):
        for user in users_data:
            if int(user["numeric_id"]) == int(numeric_id):
                return user

    # Dict formatında kontrol (eski yapı)
    elif isinstance(users_data, dict):
        for user in users_data.values():
            if int(user["numeric_id"]) == int(numeric_id):
                return user

    return None


# --- Yüz Tespit ve Kırpma (MediaPipe Tasks API) ---
def detect_face_and_crop(frame):
    """MediaPipe Tasks API ile yüz tespit eder ve kırpar."""
    if face_detector is None or not init_success:
        # Fallback: Haar Cascade kullan
        return detect_face_and_crop_haar(frame)

    try:
        # MediaPipe Image oluştur
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Yüz tespiti (video mode için detect_for_video kullan)
        detection_result = face_detector.detect_for_video(mp_image, _get_next_timestamp())

        if not detection_result.detections:
            return None, None

        h, w, _ = frame.shape
        best_face = None
        best_area = 0

        for detection in detection_result.detections:
            bbox = detection.bounding_box

            # Bounding box koordinatları
            x1 = max(int(bbox.origin_x), 0)
            y1 = max(int(bbox.origin_y), 0)
            bw = int(bbox.width)
            bh = int(bbox.height)
            x2 = min(x1 + bw, w)
            y2 = min(y1 + bh, h)

            # Standart padding (tutarlılık için)
            pad_x = int(bw * 0.20)
            pad_y = int(bh * 0.20)

            x1 = max(x1 - pad_x, 0)
            y1 = max(y1 - pad_y, 0)
            x2 = min(x2 + pad_x, w)
            y2 = min(y2 + pad_y, h)

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

    except Exception as e:
        print(f"MediaPipe detection hatası: {e}")
        # Fallback: Haar Cascade kullan
        return detect_face_and_crop_haar(frame)


# --- Haar Cascade Fallback ---
def detect_face_and_crop_haar(frame):
    """Haar cascade ile yüz tespit eder ve kırpar (fallback)."""
    # Haar cascade'i başlat
    if not hasattr(detect_face_and_crop_haar, 'cascade_loaded'):
        detect_face_and_crop_haar.cascade_loaded = False
        local_cascade = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
        if os.path.exists(local_cascade):
            detect_face_and_crop_haar.cascade = cv2.CascadeClassifier(local_cascade)
            detect_face_and_crop_haar.cascade_loaded = detect_face_and_crop_haar.cascade.load(local_cascade)
        else:
            system_cascade = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(system_cascade):
                detect_face_and_crop_haar.cascade = cv2.CascadeClassifier(system_cascade)
                detect_face_and_crop_haar.cascade_loaded = detect_face_and_crop_haar.cascade.load(system_cascade)

    if not detect_face_and_crop_haar.cascade_loaded:
        # Cascade yüklenemezse basit yöntem kullan
        h, w = frame.shape[:2]
        margin = int(min(w, h) * 0.15)
        x1, y1 = margin, margin
        x2, y2 = w - margin, h - margin
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None
        return (x1, y1, x2, y2), crop

    # Daha hızlı tespit için gri tonlama
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Eğitim için daha toleranslı parametreler
    faces = detect_face_and_crop_haar.cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(40, 40),
        maxSize=(400, 400)
    )

    if len(faces) == 0:
        return None, None

    h, w, _ = frame.shape
    best_face = None
    best_area = 0

    for (x, y, w_face, h_face) in faces:
        area = w_face * h_face
        if area > best_area:
            best_area = area
            # Eğitim için daha büyük padding
            pad_x = int(w_face * 0.2)
            pad_y = int(h_face * 0.2)

            x1 = max(x - pad_x, 0)
            y1 = max(y - pad_y, 0)
            x2 = min(x + w_face + pad_x, w)
            y2 = min(y + h_face + pad_y, h)

            best_face = (x1, y1, x2, y2)

    if best_face is None:
        return None, None

    x1, y1, x2, y2 = best_face
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None, None

    return best_face, crop


# --- Görüntü Ön İşleme (Mobil Uyumlu) ---
def preprocess_face(face_crop):
    """Yüz görüntüsünü LBPH için mobil uyumlu şekilde optimize eder."""
    # 1. Gri tonlama
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    # 2. Hafif histogram eşitleme (CLAHE - mobil uyumlu)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))  # Biraz daha güçlü kontrast
    equalized = clahe.apply(gray)

    # 3. Boyutlandırma (LBPH için standart boyut - mobil uyumlu)
    resized = cv2.resize(equalized, (160, 160))  # Daha küçük boyut - performans için

    # 4. Hafif bulanıklaştırma (gürültü azaltma - mobil uyumlu)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)

    return blurred


# --- Eğitim için Görüntü Yükleme ---
def load_and_crop_image_for_training(img_path):
    """Eğitim için bir görüntü yükler ve işler (MediaPipe Image mode ile)."""
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Eğer görüntü zaten önceden preprocess edilmiş (gri, 160x160), tekrar etme
    if img.ndim == 2 and img.shape == (160, 160):
        return img

    # Eğitim için Image mode detector kullan
    if face_detector_image is not None and init_success:
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            detection_result = face_detector_image.detect(mp_image)

            if detection_result.detections:
                # MediaPipe ile tespit başarılı
                h, w, _ = img.shape
                best_face = None
                best_area = 0

                for detection in detection_result.detections:
                    bbox = detection.bounding_box
                    x1 = max(int(bbox.origin_x), 0)
                    y1 = max(int(bbox.origin_y), 0)
                    bw = int(bbox.width)
                    bh = int(bbox.height)
                    x2 = min(x1 + bw, w)
                    y2 = min(y1 + bh, h)

                    area = (x2 - x1) * (y2 - y1)
                    if area > best_area:
                        best_area = area
                        # Eğitim için daha büyük padding
                        pad_x = int(bw * 0.2)
                        pad_y = int(bh * 0.2)
                        x1 = max(x1 - pad_x, 0)
                        y1 = max(y1 - pad_y, 0)
                        x2 = min(x2 + pad_x, w)
                        y2 = min(y2 + pad_y, h)
                        best_face = (x1, y1, x2, y2)

                if best_face is not None:
                    x1, y1, x2, y2 = best_face
                    face_crop = img[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        return preprocess_face(face_crop)
        except Exception as e:
            print(f"MediaPipe eğitim yüz işleme hatası: {e}")

    # Fallback: Haar Cascade kullan
    _, face_crop = detect_face_and_crop_haar(img)

    if face_crop is None:
        # Yüz tespit edilemedi - eğitim için tüm görüntüyü işle
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 160))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            gray = clahe.apply(gray)
            return gray
        except Exception:
            return None

    processed = preprocess_face(face_crop)
    # Eğitim için boyut kontrolü
    if processed.shape != (160, 160):
        processed = cv2.resize(processed, (160, 160))

    return processed


# --- Poz Tahmini (MediaPipe Tasks API) ---
def estimate_pose_label(frame):
    """MediaPipe Tasks API ile yüz pozunu tahmin eder."""
    if face_landmarker is None or not init_success:
        return "front", {"yaw": 0.0, "pitch": 0.0}

    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        landmark_result = face_landmarker.detect_for_video(mp_image, _get_next_timestamp())

        if not landmark_result.face_landmarks:
            return "unknown", {"yaw": 0.0, "pitch": 0.0}

        landmarks = landmark_result.face_landmarks[0]
        h, w, _ = frame.shape

        # Önemli noktaları al
        nose = landmarks[1]  # Nose tip
        left_cheek = landmarks[234]  # Left cheek
        right_cheek = landmarks[454]  # Right cheek
        forehead = landmarks[10]  # Forehead center
        chin = landmarks[152]  # Chin

        nose_x = nose.x * w
        nose_y = nose.y * h
        left_x = left_cheek.x * w
        right_x = right_cheek.x * w
        forehead_y = forehead.y * h
        chin_y = chin.y * h

        # Yüz merkezi ve boyutları
        face_center_x = (left_x + right_x) / 2.0
        face_width = max(abs(right_x - left_x), 1.0)
        face_center_y = (forehead_y + chin_y) / 2.0
        face_height = max(abs(chin_y - forehead_y), 1.0)

        # Yaw ve pitch hesapla
        yaw_ratio = (nose_x - face_center_x) / face_width
        pitch_ratio = (nose_y - face_center_y) / face_height

        yaw_deg = yaw_ratio * 45  # Daha küçük çarpan - daha toleranslı
        pitch_deg = pitch_ratio * 45

        # Poz belirleme
        if yaw_deg < -10:
            pose = "right"
        elif yaw_deg > 10:
            pose = "left"
        elif pitch_deg < -10:
            pose = "up"
        elif pitch_deg > 10:
            pose = "down"
        else:
            pose = "front"

        return pose, {"yaw": yaw_deg, "pitch": pitch_deg}

    except Exception as e:
        print(f"MediaPipe pose estimation hatası: {e}")
        return "unknown", {"yaw": 0.0, "pitch": 0.0}


# --- Yüz Doğrulama (El/Yanlış Pozitif Filtreleme) ---
def validate_real_face(frame, face_box):
    """
    Tespit edilen bölgenin gerçekten yüz içerip içermediğini
    MediaPipe face landmarks ile doğrular. El, nesne vb. yanlış
    pozitifleri filtreler.
    """
    if face_landmarker is None or not init_success:
        return True  # Landmark yoksa güvenli tarafa geç

    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmark_result = face_landmarker.detect_for_video(mp_image, _get_next_timestamp())

        if not landmark_result.face_landmarks:
            return False  # Hiç landmark yoksa yüz değildir

        # En azından temel yüz hatları var mı kontrol et
        landmarks = landmark_result.face_landmarks[0]
        if len(landmarks) < 50:
            return False  # Yetersiz landmark = muhtemelen yüz değil

        # Geometrik oran kontrolü (yüz en-boy oranı yaklaşık 0.4-1.8 arası olmalı)
        x1, y1, x2, y2 = face_box
        face_w = x2 - x1
        face_h = y2 - y1
        if face_h == 0:
            return False
        aspect_ratio = face_w / face_h
        if not (0.4 <= aspect_ratio <= 1.8):
            return False  # Yanlış oran = el veya nesne

        return True
    except Exception:
        return True  # Hata durumunda güvenli tarafa geç


# --- Görüntü Kalitesi Ölçümleri ---
def measure_blur(face_bgr):
    """Bulanıklık seviyesini ölçer (Laplacian varyansı)."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def measure_brightness(face_bgr):
    """Parlaklık seviyesini ölçer."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def advanced_preprocess_face_crop(face_crop, target_size=(160, 160), blur_threshold=None):
    """Kameradan gelen yüz kırpımına uygulanan kapsamlı ön işleme boru hattı."""
    if blur_threshold is None:
        blur_threshold = BLUR_THRESHOLD

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blurry = blur_variance < blur_threshold

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    denoised = cv2.bilateralFilter(equalized, d=5, sigmaColor=75, sigmaSpace=75)
    processed = cv2.resize(denoised, target_size)

    return processed, blur_variance, brightness, is_blurry


def create_preprocessing_comparison(raw_face, processed_face, window_height=240):
    """Ham ve işlenmiş görüntüyü yan yana gösteren karşılaştırma görüntüsü oluşturur."""
    if processed_face.ndim == 2:
        processed_display = cv2.cvtColor(processed_face, cv2.COLOR_GRAY2BGR)
    else:
        processed_display = processed_face.copy()

    raw_resized = cv2.resize(raw_face, (window_height * raw_face.shape[1] // raw_face.shape[0], window_height))
    proc_resized = cv2.resize(processed_display, (window_height * processed_display.shape[1] // processed_display.shape[0], window_height))

    combined = np.hstack([raw_resized, proc_resized])
    cv2.putText(combined, 'Raw', (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(combined, 'Processed', (raw_resized.shape[1] + 20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return combined


def get_five_point_face_landmarks(frame):
    """MediaPipe yüz landmarklarından 5 temel noktayı alır."""
    if face_landmarker is None or not init_success:
        return None

    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = face_landmarker.detect_for_video(mp_image, _get_next_timestamp())
        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]
        h, w = frame.shape[:2]

        left_eye = np.mean([[lm[33].x * w, lm[33].y * h], [lm[133].x * w, lm[133].y * h]], axis=0)
        right_eye = np.mean([[lm[362].x * w, lm[362].y * h], [lm[263].x * w, lm[263].y * h]], axis=0)
        nose_tip = np.array([lm[1].x * w, lm[1].y * h], dtype=np.float32)
        mouth_left = np.array([lm[61].x * w, lm[61].y * h], dtype=np.float32)
        mouth_right = np.array([lm[291].x * w, lm[291].y * h], dtype=np.float32)

        return {
            'left_eye': left_eye.astype(np.float32),
            'right_eye': right_eye.astype(np.float32),
            'nose_tip': nose_tip,
            'mouth_left': mouth_left,
            'mouth_right': mouth_right,
        }
    except Exception:
        return None


def align_face(frame, output_size=(160, 160)):
    """Yüzü 5 noktaya göre hizalar ve kırpar."""
    landmarks = get_five_point_face_landmarks(frame)
    if landmarks is None:
        return None

    src = np.vstack([
        landmarks['left_eye'],
        landmarks['right_eye'],
        landmarks['nose_tip'],
        landmarks['mouth_left'],
        landmarks['mouth_right'],
    ]).astype(np.float32)

    dst = np.array([
        [output_size[0] * 0.30, output_size[1] * 0.35],
        [output_size[0] * 0.70, output_size[1] * 0.35],
        [output_size[0] * 0.50, output_size[1] * 0.55],
        [output_size[0] * 0.32, output_size[1] * 0.78],
        [output_size[0] * 0.68, output_size[1] * 0.78],
    ], dtype=np.float32)

    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:
        matrix = cv2.getAffineTransform(src[:3], dst[:3])

    aligned = cv2.warpAffine(frame, matrix, output_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return aligned


def draw_alignment_comparison(raw_face, aligned_face, window_name='Alignment Comparison'):
    """Ham yüz ve hizalanmış yüzü yan yana gösterir."""
    if aligned_face is None:
        return

    aligned_resized = cv2.resize(aligned_face, (raw_face.shape[1], raw_face.shape[0]))
    comparison = np.hstack([raw_face, aligned_resized])
    cv2.putText(comparison, 'Raw Face', (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(comparison, 'Aligned Face', (raw_face.shape[1] + 20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow(window_name, comparison)


# --- Durum Paneli Çizimi ---
def draw_status_panel(frame, panel_data):
    """Ekrana bilgi paneli çizer."""
    cv2.rectangle(frame, (10, 10), (640, 305), (40, 40, 40), -1)
    cv2.rectangle(frame, (10, 10), (640, 305), (0, 255, 255), 2)

    lines = [
        f"Kamera durumu: {panel_data.get('camera_status', '-')}",
        f"Yuz algilama: {panel_data.get('face_detection', '-')}",
        f"Landmark analizi: {panel_data.get('landmark_analysis', '-')}",
        f"Yuz yonu: {panel_data.get('pose', '-')}",
        f"Taninan kullanici: {panel_data.get('recognized_user', '-')}",
        f"Kullanici ID: {panel_data.get('user_id', '-')}",
        f"Guven skoru: %{panel_data.get('confidence_score', 0)}",
        f"Netlik skoru: {panel_data.get('blur_score', 0)}",
        f"Aydinlik: {panel_data.get('brightness', 0)}",
        f"Sonraki adim: {panel_data.get('next_step', '-')}"
    ]

    y = 38
    for line in lines:
        cv2.putText(
            frame, line, (25, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 2, cv2.LINE_AA
        )
        y += 26