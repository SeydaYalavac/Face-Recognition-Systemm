import os
import json
import cv2
import numpy as np
import mediapipe as mp

from config import USERS_JSON, YUZZ_DIR

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
    os.makedirs(YUZZ_DIR, exist_ok=True)
    if not os.path.exists(USERS_JSON):
        with open(USERS_JSON, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)


def load_users():
    ensure_paths()
    with open(USERS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def find_user_by_numeric_id(numeric_id):
    users_data = load_users()
    # users_data.values() kullanarak sözlüğün içindeki gerçek verilere ulaşıyoruz
    for user in users_data.values():
        if int(user["numeric_id"]) == int(numeric_id):
            return user
    return None


def detect_face_and_crop(frame):
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


def preprocess_face(face_crop):
    # Griye çevir
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    # --- NETLEŞTİRME VE DENGELEME ---
    # 1. Işığı her yüz için standart hale getir
    equalized = cv2.equalizeHist(gray)

    # 2. Gürültüyü azalt (Küçük lekeleri siler)
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)

    return blurred
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (200, 200))

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Parlaklik dengeleme: cok karanliksa kontrol amacli aydinlat
    mean_val = float(gray.mean())
    if mean_val < 90:
        gray = cv2.convertScaleAbs(gray, alpha=1.4, beta=20)
    elif mean_val < 110:
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def load_and_crop_image_for_training(img_path):
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
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def measure_brightness(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def draw_status_panel(frame, panel_data):
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
    import numpy as np

    def get_face_embedding(face_crop):
        """
        Yüzün karakteristik özelliklerini sayısal bir vektöre (embedding) çevirir.
        Şu anki LBPH mantığını bu vektör karşılaştırmasına taşıyacağız.
        """
        # Görüntüyü standart boyuta getir (Örn: 160x160 FaceNet standardı)
        face_resized = cv2.resize(face_crop, (160, 160))

        # Piksel değerlerini normalize et (0-1 arasına çek)
        face_normalized = face_resized.astype('float32') / 255.0

        # Not: Burada profesyonel bir model (FaceNet.h5 gibi) yüklü olmalıdır.
        # Şimdilik mevcut sistemine 'geometrik embedding' mantığı ekliyoruz.
        return face_normalized.flatten()  # Bu senin 'yüz imzan' olacak