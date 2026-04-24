import os
import cv2
from collections import Counter, deque

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    TRAINER_PATH, WINDOW_NAME_RECOGNIZE,
    LBPH_STRICT_THRESHOLD, LBPH_SOFT_THRESHOLD,
    MIN_FACE_SIZE, BLUR_THRESHOLD, BRIGHTNESS_THRESHOLD,
    USERS_JSON,
    MIN_DISPLAY_SCORE_TO_ACCEPT, DOMINANCE_DIFF,
    HISTORY_WINDOW, PREDICT_EVERY_N_FRAMES
)
from face_utils import (
    detect_face_and_crop, preprocess_face,
    estimate_pose_label, find_user_by_numeric_id,
    draw_status_panel, measure_blur, measure_brightness,
    check_device_stability, validate_real_face
)
from output_formatter import build_output

# --- Sabitler ---
BRIGHTNESS_MIN = BRIGHTNESS_THRESHOLD
POSE_EVERY_N_FRAMES = 10  # Poz tahmini
FACE_DETECT_EVERY_N_FRAMES = 2  # Yüz tespiti her 2 frame'de bir

MIN_REPEATS_FRONT = 2
MIN_REPEATS_PROFILE = 2
MIN_REPEATS_UPDOWN = 2


def get_registered_names():
    """users.json'dan kayıtlı isimleri dinamik olarak alır (küçük harf)."""
    import json
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
    """LBPH skorunu kullanıcı dostu güven skoruna çevirir.
    LBPH mesafesi ne kadar düşükse o kadar iyidir.
    Bu dönüşüm mobilde daha yüksek soft skorları destekler.
    """
    score = 100.0 - (conf * 0.4)
    score = max(0.0, min(100.0, score))
    return round(score, 2)


def most_common_value(items):
    """Liste içindeki en sık değeri döndürür."""
    if not items:
        return None
    counter = Counter(items)
    return counter.most_common(1)[0][0]


def get_top_two_counts(items):
    """İlk iki en sık değeri ve sayılarını döndürür."""
    counter = Counter(items)
    most_common = counter.most_common(2)

    if not most_common:
        return None, 0, 0

    if len(most_common) == 1:
        return most_common[0][0], most_common[0][1], 0

    return most_common[0][0], most_common[0][1], most_common[1][1]


def recognize_live(bellek, sensor_data=None, lux_val=100):
    """
    Canlı yüz tanıma fonksiyonu.

    Args:
        bellek: Bellek sistemi nesnesi
        sensor_data: İvmeölçer verileri (opsiyonel)
        lux_val: Işık seviyesi (opsiyonel)
    """
    if not os.path.exists(TRAINER_PATH):
        print("Egitilmis model bulunamadi. Once modeli egitin.")
        return

    # Kayıtlı kullanıcı isimlerini dinamik olarak al
    registered_names = get_registered_names()

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("HATA: Kamera acilamadi!")
        return

    print("Canli tanima basladi. Cikmak icin Q veya ESC tusuna basin.")
    print("Son ciktiyi terminalde gormek icin P tusuna basin.")

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
                print("Kamera okunamadi.")
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

            # Sarsıntı kontrolü
            if sensor_data is not None and not check_device_stability(sensor_data):
                panel_data["next_step"] = "sarsinti algilandi"
                draw_status_panel(frame, panel_data)
                cv2.imshow(WINDOW_NAME_RECOGNIZE, frame)
                if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                    break
                continue

            # Işık kontrolü
            if lux_val < 30:
                panel_data["next_step"] = "karanlik: isik yetersiz"
                draw_status_panel(frame, panel_data)
                cv2.imshow(WINDOW_NAME_RECOGNIZE, frame)
                if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                    break
                continue

            # --- Yüz Tespit ---
            if frame_counter % FACE_DETECT_EVERY_N_FRAMES == 0:
                face_box, face_crop = detect_face_and_crop(frame)
                last_face_box = face_box
                last_face_crop = face_crop
            else:
                face_box = last_face_box
                face_crop = last_face_crop

            if face_box is not None:
                # --- Yüz Doğrulama (El/Yanlış Pozitif Filtreleme) ---
                if not validate_real_face(frame, face_box):
                    panel_data["face_detection"] = "gecersiz (yuz degil)"
                    panel_data["next_step"] = "yuz dogrulanamadi"
                    draw_status_panel(frame, panel_data)
                    cv2.imshow(WINDOW_NAME_RECOGNIZE, frame)
                    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                        break
                    continue

                x1, y1, x2, y2 = face_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                panel_data["face_detection"] = "basarili"

                if (x2 - x1) >= MIN_FACE_SIZE and (y2 - y1) >= MIN_FACE_SIZE:
                    # Poz tahmini
                    if frame_counter % POSE_EVERY_N_FRAMES == 0:
                        pose, _ = estimate_pose_label(frame)
                        last_pose = pose
                    else:
                        pose = last_pose

                    panel_data["landmark_analysis"] = "tamamlandi"
                    panel_data["pose"] = pose

                    blur_value = measure_blur(face_crop)
                    bright_value = measure_brightness(face_crop)
                    panel_data["blur_score"] = round(blur_value, 2)
                    panel_data["brightness"] = round(bright_value, 2)

                    # Poza göre eşik ayarı
                    if pose in ("right", "left"):
                        strict_threshold = LBPH_STRICT_THRESHOLD + 3
                        soft_threshold = LBPH_SOFT_THRESHOLD + 3
                        min_required_repeats = MIN_REPEATS_PROFILE
                    elif pose in ("up", "down"):
                        strict_threshold = LBPH_STRICT_THRESHOLD + 2
                        soft_threshold = LBPH_SOFT_THRESHOLD + 2
                        min_required_repeats = MIN_REPEATS_UPDOWN
                    else:
                        strict_threshold = LBPH_STRICT_THRESHOLD
                        soft_threshold = LBPH_SOFT_THRESHOLD
                        min_required_repeats = MIN_REPEATS_FRONT

                    # Karanlık reddetme
                    if bright_value < BRIGHTNESS_MIN:
                        panel_data["recognized_user"] = "unknown"
                        panel_data["user_id"] = "N/A"
                        panel_data["confidence_score"] = 0
                        panel_data["next_step"] = "goruntu cok karanlik, kabul edilmedi"

                        cv2.putText(
                            frame, "unknown | dusuk isik",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2
                        )

                        last_output = build_output(
                            user_id="N/A", name="unknown", recognized=False,
                            confidence=0, pose=pose,
                            next_step="goruntu cok karanlik, kabul edilmedi"
                        )

                    # Bulanıklık reddetme
                    elif blur_value < BLUR_THRESHOLD:
                        panel_data["next_step"] = "goruntu bulanik, tekrar deneyin"

                    else:
                        # Tanıma işlemi
                        if frame_counter % PREDICT_EVERY_N_FRAMES == 0:
                            processed = preprocess_face(face_crop)
                            label, conf = recognizer.predict(processed)
                            display_score = conf_to_display_score(conf)
                            user = find_user_by_numeric_id(label) if label is not None else None

                            recent_raw_conf.append(conf)

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

                        # Ortalama skor HESAPLAMA: sadece strict/soft değerler
                        strict_soft_scores = [s for s, v in zip(recent_scores, recent_valid) if v in ("strict", "soft")]
                        if strict_soft_scores:
                            avg_score = round(sum(strict_soft_scores) / len(strict_soft_scores), 2)
                        else:
                            avg_score = 0.0

                        # Raw LBPH yüzey mesafesi ortalaması
                        valid_raw = [c for c, v in zip(recent_raw_conf, recent_valid) if v in ("strict", "soft")]
                        avg_raw_conf = round(sum(valid_raw) / len(valid_raw), 2) if valid_raw else None

                        # Sadece geçerli tahminlerdeki isimleri say
                        valid_names = [n for n, v in zip(recent_names, recent_valid) if v in ("strict", "soft")]
                        valid_ids   = [i for i, v in zip(recent_ids, recent_valid) if v in ("strict", "soft")]

                        if valid_names:
                            stable_name, top_count, second_count = get_top_two_counts(valid_names)
                            stable_id = most_common_value(valid_ids)
                        else:
                            stable_name, top_count, second_count = None, 0, 0
                            stable_id = None

                        accept = False

                        # Kayıtlı kişi mi diye kontrol et (küçük harf)
                        stable_name_lower = (stable_name or "").strip().lower()
                        is_registered_user = stable_name_lower in registered_names

                        if is_registered_user and avg_raw_conf is not None:
                            score_threshold = MIN_DISPLAY_SCORE_TO_ACCEPT
                            strict_count = sum(1 for v in recent_valid if v == "strict")
                            soft_count = sum(1 for v in recent_valid if v == "soft")
                            total_valid = strict_count + soft_count

                            if total_valid >= min_required_repeats:
                                if strict_count >= 2 and avg_raw_conf <= strict_threshold and avg_score >= score_threshold and (top_count - second_count) >= DOMINANCE_DIFF:
                                    accept = True
                                elif strict_count >= 1 and soft_count >= 2 and avg_raw_conf <= soft_threshold and avg_score >= max(score_threshold - 5, 35) and (top_count - second_count) >= DOMINANCE_DIFF:
                                    accept = True
                                elif soft_count >= 3 and avg_raw_conf <= soft_threshold and avg_score >= max(score_threshold - 10, 35) and (top_count - second_count) >= DOMINANCE_DIFF:
                                    accept = True
                            else:
                                accept = False
                        else:
                            accept = False

                        if accept:
                            panel_data["recognized_user"] = stable_name
                            panel_data["user_id"] = stable_id
                            panel_data["confidence_score"] = avg_score
                            panel_data["next_step"] = "bellek kontrolune gonderildi"

                            cv2.putText(
                                frame, f"{stable_name} | {stable_id} | %{avg_score:.1f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 255, 0), 2
                            )

                            last_output = build_output(
                                user_id=stable_id, name=stable_name, recognized=True,
                                confidence=avg_score, pose=pose,
                                next_step="bellek kontrolune gonderildi"
                            )
                        else:
                            panel_data["recognized_user"] = "unknown"
                            panel_data["user_id"] = "N/A"
                            panel_data["confidence_score"] = avg_score
                            panel_data["next_step"] = "kararsiz, bellek kontrolune gonderilmedi"

                            cv2.putText(
                                frame, f"unknown | %{avg_score:.1f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 0, 255), 2
                            )

                            last_output = build_output(
                                user_id="N/A", name="unknown", recognized=False,
                                confidence=avg_score, pose=pose,
                                next_step=panel_data["next_step"]
                            )

            draw_status_panel(frame, panel_data)
            cv2.imshow(WINDOW_NAME_RECOGNIZE, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("p") and last_output is not None:
                print("\nSon cikti:")
                print(last_output)

            if key == ord("q") or key == 27:
                break

    except Exception as e:
        print(f"Tanimada kritik hata: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Kamera güvenli bir şekilde kapatildi.")

