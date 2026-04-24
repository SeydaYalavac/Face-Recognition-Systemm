import os
import json
import cv2
from collections import Counter, deque

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    TRAINER_PATH, WINDOW_NAME_RECOGNIZE,
    LBPH_STRICT_THRESHOLD, LBPH_SOFT_THRESHOLD,
    MIN_FACE_SIZE, BLUR_THRESHOLD, BRIGHTNESS_THRESHOLD,
    USERS_JSON,
    MIN_DISPLAY_SCORE_TO_ACCEPT, DOMINANCE_DIFF,
    HISTORY_WINDOW, PREDICT_EVERY_N_FRAMES,
    TEST_MODE, TEST_LBPH_SOFT_THRESHOLD, TEST_MIN_DISPLAY_SCORE_TO_ACCEPT,
    TEST_BLUR_THRESHOLD, TEST_BRIGHTNESS_THRESHOLD, TEST_DOMINANCE_DIFF
)
from face_utils import (
    detect_face_and_crop, preprocess_face, align_face,
    advanced_preprocess_face_crop, create_preprocessing_comparison,
    estimate_pose_label, find_user_by_numeric_id,
    draw_status_panel, measure_blur, measure_brightness,
    check_device_stability, validate_real_face
)
from output_formatter import build_output

BRIGHTNESS_MIN = 40
POSE_EVERY_N_FRAMES = 10
FACE_DETECT_EVERY_N_FRAMES = 2

MIN_REPEATS_FRONT = 1
MIN_REPEATS_PROFILE = 1
MIN_REPEATS_UPDOWN = 1

# Test modu aktifse esnek eşikleri kullan
if TEST_MODE:
    ACTIVE_LBPH_SOFT_THRESHOLD = TEST_LBPH_SOFT_THRESHOLD
    ACTIVE_MIN_DISPLAY_SCORE = TEST_MIN_DISPLAY_SCORE_TO_ACCEPT
    ACTIVE_BLUR_THRESHOLD = TEST_BLUR_THRESHOLD
    ACTIVE_BRIGHTNESS_THRESHOLD = TEST_BRIGHTNESS_THRESHOLD
    ACTIVE_DOMINANCE_DIFF = TEST_DOMINANCE_DIFF
else:
    ACTIVE_LBPH_SOFT_THRESHOLD = LBPH_SOFT_THRESHOLD
    ACTIVE_MIN_DISPLAY_SCORE = MIN_DISPLAY_SCORE_TO_ACCEPT
    ACTIVE_BLUR_THRESHOLD = BLUR_THRESHOLD
    ACTIVE_BRIGHTNESS_THRESHOLD = BRIGHTNESS_THRESHOLD
    ACTIVE_DOMINANCE_DIFF = DOMINANCE_DIFF


def get_registered_names():
    try:
        with open(USERS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return [u.get("name", "").strip().lower() for u in data.values()]
        elif isinstance(data, list):
            return [u.get("name", "").strip().lower() for u in data]
    except Exception:
        pass
    return []


def conf_to_display_score(conf: float) -> float:
    """LBPH mesafesini tersine çevirerek 0-100 arası skor üretir.
    Düşük mesafe = yüksek güven. Test modunda normalize edilir."""
    if TEST_MODE:
        # Test: conf 150-350 arası tipik, 150->100%, 350->0%
        score = (350.0 - conf) / 2.0
    else:
        score = 100.0 - conf
    return max(0.0, min(100.0, round(score, 2)))


def most_common_value(items):
    if not items:
        return None
    return Counter(items).most_common(1)[0][0]


def get_top_two_counts(items):
    counter = Counter(items)
    most_common = counter.most_common(2)
    if not most_common:
        return None, 0, 0
    if len(most_common) == 1:
        return most_common[0][0], most_common[0][1], 0
    return most_common[0][0], most_common[0][1], most_common[1][1]


def debug_verify_labels():
    """trainer.yml içindeki label'ları okuyup users.json ile karşılaştırır."""
    if not os.path.exists(TRAINER_PATH):
        print("[DEBUG] trainer.yml bulunamadi.")
        return
    try:
        fs = cv2.FileStorage(TRAINER_PATH, cv2.FILE_STORAGE_READ)
        labels_node = fs.getNode("labels")
        model_labels = sorted(set(int(labels_node.at(i).real()) for i in range(labels_node.size())))
        fs.release()

        with open(USERS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            json_ids = sorted(int(u.get("numeric_id", -1)) for u in data.values())
        else:
            json_ids = sorted(int(u.get("numeric_id", -1)) for u in data)

        print(f"[DEBUG] Model labels (trainer.yml): {model_labels}")
        print(f"[DEBUG] Users numeric_id (users.json): {json_ids}")
        if model_labels == json_ids:
            print("[DEBUG] ✅ Label eslesme tamam.")
        else:
            print("[DEBUG] ⚠️ Label eslesme hatasi! Fark: "
                  f"{set(model_labels).symmetric_difference(set(json_ids))}")
    except Exception as e:
        print(f"[DEBUG] Label dogrulama hatasi: {e}")


def recognize_live(bellek, sensor_data=None, lux_val=100):
    if not os.path.exists(TRAINER_PATH):
        print("Model bulunamadi.")
        return

    registered_names = get_registered_names()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)

    # --- DEBUG: Label eslesme kontrolu ---
    debug_verify_labels()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Kamera acilamadi!")
        return

    print("Tanima basladi. Q/ESC: cikis, P: son cikti")
    if TEST_MODE:
        print("⚠️ TEST MODU AKTIF - Filtreler esnetildi.")

    frame_counter = 0
    last_output = None
    last_pose = "front"
    last_face_box = None
    last_face_crop = None

    recent_names = deque(maxlen=HISTORY_WINDOW)
    recent_ids = deque(maxlen=HISTORY_WINDOW)
    recent_scores = deque(maxlen=HISTORY_WINDOW)
    recent_raw_conf = deque(maxlen=HISTORY_WINDOW)
    recent_valid = deque(maxlen=HISTORY_WINDOW)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_counter += 1

            panel_data = {
                "camera_status": "aktif",
                "face_detection": "bekleniyor",
                "landmark_analysis": "bekleniyor",
                "pose": "unknown",
                "recognized_user": "-",
                "user_id": "-",
                "confidence_score": 0,
                "blur_score": 0,
                "brightness": 0,
                "next_step": "yuz bekleniyor"
            }

            # Test modu uyarısı
            if TEST_MODE:
                cv2.putText(frame, "TEST MODU AKTIF", (10, FRAME_HEIGHT - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Sarsinti kontrolu
            if sensor_data is not None and not check_device_stability(sensor_data):
                panel_data["next_step"] = "sarsinti"
                draw_status_panel(frame, panel_data)
                cv2.imshow(WINDOW_NAME_RECOGNIZE, frame)
                if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                    break
                continue

            # Isik kontrolu (test modunda esnet)
            light_limit = 20 if not TEST_MODE else 5
            if lux_val < light_limit:
                panel_data["next_step"] = "isik yetersiz"
                draw_status_panel(frame, panel_data)
                cv2.imshow(WINDOW_NAME_RECOGNIZE, frame)
                if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                    break
                continue

            # Yuz tespit
            if frame_counter % FACE_DETECT_EVERY_N_FRAMES == 0:
                face_box, face_crop = detect_face_and_crop(frame)
                last_face_box = face_box
                last_face_crop = face_crop
            else:
                face_box = last_face_box
                face_crop = last_face_crop

            if face_box is not None:
                x1, y1, x2, y2 = face_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                panel_data["face_detection"] = "basarili"

                if (x2 - x1) >= MIN_FACE_SIZE and (y2 - y1) >= MIN_FACE_SIZE:
                    if frame_counter % POSE_EVERY_N_FRAMES == 0:
                        pose, _ = estimate_pose_label(frame)
                        last_pose = pose
                    else:
                        pose = last_pose

                    panel_data["landmark_analysis"] = "tamamlandi"
                    panel_data["pose"] = pose

                    # Egitimle ayni on isleme kullan (align_face devre disi)
                    processed = preprocess_face(face_crop)

                    # Blur ve brightness ayri olc
                    blur_value = measure_blur(face_crop)
                    bright_value = measure_brightness(face_crop)
                    is_blurry = blur_value < ACTIVE_BLUR_THRESHOLD
                    panel_data["blur_score"] = round(blur_value, 2)
                    panel_data["brightness"] = round(bright_value, 2)

                    comparison = create_preprocessing_comparison(face_crop, processed)
                    cv2.imshow("Preprocess Comparison", comparison)

                    # Poza gore esik (test modunda daha toleransli)
                    base_strict = LBPH_STRICT_THRESHOLD + (2 if not TEST_MODE else 15)
                    base_soft = ACTIVE_LBPH_SOFT_THRESHOLD + (2 if not TEST_MODE else 15)

                    if pose in ("right", "left"):
                        strict_threshold = base_strict + 8
                        soft_threshold = base_soft + 8
                    elif pose in ("up", "down"):
                        strict_threshold = base_strict + 6
                        soft_threshold = base_soft + 6
                    else:
                        strict_threshold = base_strict
                        soft_threshold = base_soft

                    min_required_repeats = 1

                    # Karanlik kontrolu (test modunda atla)
                    bright_limit = 35 if not TEST_MODE else 5
                    if bright_value < bright_limit and not TEST_MODE:
                        panel_data["next_step"] = "cok karanlik"
                        cv2.putText(frame, "unknown | dusuk isik", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        last_output = build_output("N/A", "unknown", False, 0, pose, "cok karanlik")

                    # Bulaniklik kontrolu (test modunda atla)
                    elif blur_value < 70 and not TEST_MODE:
                        panel_data["next_step"] = "goruntu bulanik"
                        cv2.putText(frame, "unknown | bulanik", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    else:
                        # Tanima
                        if frame_counter % PREDICT_EVERY_N_FRAMES == 0:
                            label, conf = recognizer.predict(processed)
                            display_score = conf_to_display_score(conf)
                            user = find_user_by_numeric_id(label) if label is not None else None

                            recent_raw_conf.append(conf)

                            user_name = user["name"] if user else "None"
                            print(f"[DEBUG] label={label}, conf={conf:.1f}, score={display_score:.1f}, user={user_name}, strict={strict_threshold:.1f}, soft={soft_threshold:.1f}")

                            if user is not None:
                                recent_names.append(user["name"])
                                recent_ids.append(user["user_id"])
                                recent_scores.append(display_score)

                                if conf <= strict_threshold:
                                    recent_valid.append("strict")
                                elif conf <= soft_threshold:
                                    recent_valid.append("soft")
                                else:
                                    recent_valid.append("weak")
                            else:
                                recent_names.append("unknown")
                                recent_ids.append("N/A")
                                recent_scores.append(display_score)
                                recent_valid.append("unknown")

                        # Skor hesaplama
                        # Kimlik belirleme (stable_name) icin SADECE strict/soft kullan
                        # weak/unknown tahminler taninmayan kisi icin yanlis isim cikmasina neden olur
                        identity_types = ("strict", "soft")
                        score_types = ("strict", "soft", "weak") if TEST_MODE else ("strict", "soft")

                        strict_soft_scores = [s for s, v in zip(recent_scores, recent_valid)
                                              if v in score_types]
                        avg_score = round(sum(strict_soft_scores) / len(strict_soft_scores), 2) \
                            if strict_soft_scores else 0.0

                        valid_raw = [c for c, v in zip(recent_raw_conf, recent_valid)
                                     if v in score_types]
                        avg_raw_conf = round(sum(valid_raw) / len(valid_raw), 2) if valid_raw else None

                        # Kimlik oylamasi: SADECE strict/soft dahil et
                        valid_names = [n for n, v in zip(recent_names, recent_valid)
                                       if v in identity_types]
                        valid_ids = [i for i, v in zip(recent_ids, recent_valid)
                                     if v in identity_types]

                        if valid_names:
                            stable_name, top_count, second_count = get_top_two_counts(valid_names)
                            stable_id = most_common_value(valid_ids)
                        else:
                            stable_name, top_count, second_count = None, 0, 0
                            stable_id = None

                        accept = False
                        stable_name_lower = (stable_name or "").strip().lower()
                        is_registered_user = stable_name_lower in registered_names

                        if is_registered_user and avg_raw_conf is not None:
                            strict_count = sum(1 for v in recent_valid if v == "strict")
                            soft_count = sum(1 for v in recent_valid if v == "soft")
                            total_valid = strict_count + soft_count

                            dominance_ok = (top_count - second_count) >= ACTIVE_DOMINANCE_DIFF
                            score_ok = avg_score >= ACTIVE_MIN_DISPLAY_SCORE
                            repeats_ok = total_valid >= min_required_repeats
                            # Taninmayan kisi onleme: ortalama raw conf soft_threshold'un altinda olmali
                            conf_ok = avg_raw_conf <= soft_threshold

                            print(f"[DEBUG] accept_check: stable={stable_name}, avg_score={avg_score}, avg_raw_conf={avg_raw_conf}, score_ok={score_ok}, repeats_ok={repeats_ok}, dominance_ok={dominance_ok}, conf_ok={conf_ok}, top={top_count}, second={second_count}")

                            if score_ok and repeats_ok and dominance_ok and conf_ok:
                                accept = True

                        if accept:
                            panel_data["recognized_user"] = stable_name
                            panel_data["user_id"] = stable_id
                            panel_data["confidence_score"] = avg_score
                            panel_data["next_step"] = "bellek kontrolune gonderildi"

                            cv2.putText(frame, f"{stable_name} | {stable_id} | %{avg_score:.1f}",
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                            last_output = build_output(stable_id, stable_name, True, avg_score, pose,
                                                       "bellek kontrolune gonderildi")
                        else:
                            panel_data["recognized_user"] = "unknown"
                            panel_data["user_id"] = "N/A"
                            panel_data["confidence_score"] = avg_score
                            panel_data["next_step"] = "kararsiz"

                            cv2.putText(frame, f"unknown | %{avg_score:.1f}",
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                            last_output = build_output("N/A", "unknown", False, avg_score, pose, "kararsiz")

            draw_status_panel(frame, panel_data)
            cv2.imshow(WINDOW_NAME_RECOGNIZE, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("p") and last_output is not None:
                print("\nSon cikti:")
                print(last_output)

            if key == ord("q") or key == 27:
                break

    except Exception as e:
        print(f"Hata: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize_live(bellek=None)
