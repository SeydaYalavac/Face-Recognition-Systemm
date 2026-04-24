import os
import cv2
import numpy as np

from config import (
    YUZZ_DIR, TRAINER_PATH,
    LBPH_RADIUS, LBPH_NEIGHBORS, LBPH_GRID_X, LBPH_GRID_Y
)
from face_utils import load_users, load_and_crop_image_for_training


def augment_face(image):
    """Eğitim için basit veri artırma teknikleri uygular."""
    augmented = []
    
    # Orijinal
    augmented.append(image)
    
    # Yatay çevirme
    flipped = cv2.flip(image, 1)
    augmented.append(flipped)
    
    # Hafif parlaklık değişimi (+/- 15)
    bright = np.clip(image.astype(np.int16) + np.random.randint(-15, 16), 0, 255).astype(np.uint8)
    augmented.append(bright)
    
    # Hafif kontrast değişimi (scale 0.9 - 1.1)
    alpha = np.random.uniform(0.9, 1.1)
    contrast = np.clip(image.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    augmented.append(contrast)
    
    return augmented


def train_lbph_model():
    users_data = load_users() # Sözlüğü al
    if not users_data:
        print("❌ Eğitilecek kullanıcı yok. users.json dosyasını kontrol edin.")
        return False

    # Dict formatını işle
    if isinstance(users_data, dict):
        users_list = list(users_data.values())
    else:
        users_list = users_data

    if not users_list:
        print("❌ Kullanıcı listesi boş.")
        return False

    faces = []
    labels = []

    for user in users_list:
        numeric_id = user.get("numeric_id")
        folder_name = user.get("folder")
        user_name = user.get("name", "Unknown")

        if not numeric_id or not folder_name:
            print(f"⚠️ Kullanıcı verisi eksik: {user}")
            continue

        folder_path = os.path.join(YUZZ_DIR, folder_name)

        if not os.path.exists(folder_path):
            print(f"⚠️ Klasör yok: {folder_path}")
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

                # Kalite kontrolü - eğitim için daha toleranslı
                if processed.shape != (160, 160):
                    processed = cv2.resize(processed, (160, 160))

                # Kontrast kontrolü - çok düşük kontrastlı görüntüleri atla
                if processed.dtype == np.uint8:
                    min_val, max_val = np.min(processed), np.max(processed)
                    if max_val - min_val < 30:  # Çok düşük kontrast
                        skipped_count += 1
                        continue

                # Veri artırma uygula
                aug_images = augment_face(processed)
                for aug_img in aug_images:
                    faces.append(aug_img)
                    labels.append(numeric_id)
                added_count += len(aug_images)

        print(f"👤 {user_name} için eğitime eklenen: {added_count}, atlanan: {skipped_count}")

    if not faces:
        print("❌ Eğitim için geçerli görüntü bulunamadı.")
        print("💡 Öneriler:")
        print("   - Kullanıcı kayıtlarını kontrol edin")
        print("   - Yüz görüntülerinin kalitesini kontrol edin")
        print("   - Haar cascade dosyasının mevcut olduğundan emin olun")
        return False

    print(f"📊 Eğitim verisi: {len(faces)} görüntü, {len(set(labels))} kullanıcı")

    # LBPH recognizer oluştur
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=LBPH_RADIUS,
        neighbors=LBPH_NEIGHBORS,
        grid_x=LBPH_GRID_X,
        grid_y=LBPH_GRID_Y
    )

    # Eğitim
    try:
        recognizer.train(faces, np.array(labels))
        recognizer.write(TRAINER_PATH)
        print(f"✅ Model eğitildi: {TRAINER_PATH}")
        print(f"📈 Toplam görüntü: {len(faces)}")
        print(f"👥 Kullanıcı sayısı: {len(set(labels))}")
        return True
    except Exception as e:
        print(f"❌ Model eğitimi başarısız: {e}")
        return False


if __name__ == "__main__":
    print("🤖 Yüz tanıma modeli eğitimi başlatılıyor...")
    success = train_lbph_model()
    if success:
        print("🎉 Eğitim başarıyla tamamlandı!")
    else:
        print("💥 Eğitim başarısız oldu!")