import os
import cv2
import numpy as np

from config import (
    YUZZ_DIR, TRAINER_PATH,
    LBPH_RADIUS, LBPH_NEIGHBORS, LBPH_GRID_X, LBPH_GRID_Y
)
from face_utils import load_users, load_and_crop_image_for_training


def train_lbph_model():
    users_data = load_users() # Sözlüğü al
    if not users_data:
        print("Egitilecek kullanici yok. users.json dosyasini kontrol edin.")
        return

    faces = []
    labels = []

    # users_data.values() kullanarak doğrudan içerideki bilgilere ulaşıyoruz
    for user in users_data.values():
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