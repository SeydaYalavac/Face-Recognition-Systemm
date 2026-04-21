import os
import json
import time
import cv2
import numpy as np

# 'configg' hatasını önlemek için 'config' olarak import ediyoruz
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    YUZZ_DIR, USERS_JSON, MIN_FACE_SIZE, WINDOW_NAME_REGISTER
)
from face_utils import (
    ensure_paths, detect_face_and_crop, preprocess_face,
    estimate_pose_label, draw_status_panel, load_users
)

# SIRALI TALİMATLAR
REQUIRED_POSES = ["front", "right", "left", "up", "down"]
PHOTOS_PER_POSE = 15


def get_next_user_number():
    users_data = load_users()
    if not users_data or len(users_data) == 0:
        return 1

    # HATA BURADAYDI: users_data.values() diyerek sözlüğün içindeki verilere ulaşıyoruz
    try:
        # Sözlükteki tüm kullanıcıların numeric_id değerlerini al ve en büyüğünü bul
        ids = [int(u["numeric_id"]) for u in users_data.values()]
        return max(ids) + 1
    except (TypeError, KeyError, AttributeError):
        # Eğer users_data bir liste ise (eski yapıdan kaldıysa) bu şekilde dene
        try:
            ids = [int(u["numeric_id"]) for u in users_data]
            return max(ids) + 1
        except:
            return 1


def save_new_user_to_json(name, numeric_id, folder_name):
    """Kullanıcıyı users.json dosyasına ekler."""
    users = load_users()

    new_user = {
        "numeric_id": numeric_id,
        "user_id": f"ID_{numeric_id:03d}",
        "name": name.strip().lower(),
        "folder": folder_name
    }

    users.append(new_user)

    with open(USERS_JSON, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)
    return new_user


def register_new_user(name):
    ensure_paths()
    numeric_id = get_next_user_number()
    folder_name = f"yuz{numeric_id}"  # Otomatik klasör ismi
    folder_path = os.path.join(YUZZ_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("HATA: Kamera acilamadi!")
        return

    pose_counts = {pose: 0 for pose in REQUIRED_POSES}
    current_pose_index = 0
    last_capture_time = 0

    print(f"Kayit basladi: {name} (Klasor: {folder_name})")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        # Hedef pozu belirle
        if current_pose_index < len(REQUIRED_POSES):
            target_pose = REQUIRED_POSES[current_pose_index]
        else:
            target_pose = "TAMAMLANDI"

        # --- TALİMAT EKRANI ---
        cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
        if target_pose != "TAMAMLANDI":
            msg = f"TALIMAT: {target_pose.upper()} BAKIN! ({pose_counts[target_pose]}/{PHOTOS_PER_POSE})"
            color = (0, 255, 255)
        else:
            msg = "KAYIT BITTI! 'Q' TUSUNA BASIN."
            color = (0, 255, 0)
        cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        face_box, face_crop = detect_face_and_crop(frame)
        if face_box is not None:
            x1, y1, x2, y2 = face_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Yüz pozunu algıla
            pose, _ = estimate_pose_label(frame)

            if target_pose != "TAMAMLANDI" and pose == target_pose:
                curr_time = time.time()
                if curr_time - last_capture_time > 0.4:
                    processed = preprocess_face(face_crop)
                    img_name = f"{target_pose}_{pose_counts[target_pose]}.jpg"
                    cv2.imwrite(os.path.join(folder_path, img_name), processed)
                    pose_counts[target_pose] += 1
                    last_capture_time = curr_time

                    if pose_counts[target_pose] >= PHOTOS_PER_POSE:
                        current_pose_index += 1

        cv2.imshow(WINDOW_NAME_REGISTER, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

    # İşlem başarıyla bittiyse JSON'a kaydet
    save_new_user_to_json(name, numeric_id, folder_name)
    print(f"Basariyla kaydedildi: {name}")