import os
import cv2
from collections import Counter, deque
from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    TRAINER_PATH, WINDOW_NAME_RECOGNIZE,
    LBPH_STRICT_THRESHOLD, LBPH_SOFT_THRESHOLD,
    MIN_FACE_SIZE, BLUR_THRESHOLD
)
from face_utils import (
    detect_face_and_crop, preprocess_face,
    estimate_pose_label, find_user_by_numeric_id,
    draw_status_panel, measure_blur, measure_brightness
)
from output_formatter import build_output


# Karanlık eşiği biraz daha esnek yapıldı
BRIGHTNESS_MIN = 65.0

# Karar penceresi
HISTORY_WINDOW = 30
PREDICT_EVERY_N_FRAMES = 2

# Kabul için gereken minimum tekrar
MIN_REPEATS_FRONT = 10
MIN_REPEATS_PROFILE = 7
MIN_REPEATS_UPDOWN = 6

# Bilinmeyenleri yanlış tanımamak için minimum skor
MIN_DISPLAY_SCORE_TO_ACCEPT = 70.0

# Karar tutarlılığı
DOMINANCE_DIFF = 8


def conf_to_display_score(conf: float) -> float:
    if conf <= 30:
        return 96.0
    elif conf <= 40:
        return 90.0
    elif conf <= 50:
        return 84.0
    elif conf <= 60:
        return 76.0
    elif conf <= 70:
        return 66.0
    elif conf <= 80:
        return 52.0
    else:
        return 35.0


def most_common_value(items):
    if not items:
        return None
    counter = Counter(items)
    return counter.most_common(1)[0][0]


def get_top_two_counts(items):
    counter = Counter(items)
    most_common = counter.most_common(2)

    if not most_common:
        return None, 0, 0

    if len(most_common) == 1:
        return most_common[0][0], most_common[0][1], 0

    return most_common[0][0], most_common[0][1], most_common[1][1]


def recognize_live(bellek):

    if not os.path.exists(TRAINER_PATH):
        print("Egitilmis model bulunamadi. Once modeli egitin.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("Canli tanima basladi. Cikmak icin Q veya ESC tusuna basin.")
    print("Son ciktiyi terminalde gormek icin P tusuna basin.")

    frame_counter = 0
    last_output = None

    recent_names = deque(maxlen=HISTORY_WINDOW)
    recent_ids = deque(maxlen=HISTORY_WINDOW)
    recent_scores = deque(maxlen=HISTORY_WINDOW)
    recent_valid = deque(maxlen=HISTORY_WINDOW)

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

        face_box, face_crop = detect_face_and_crop(frame)

        if face_box is not None:
            x1, y1, x2, y2 = face_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            panel_data["face_detection"] = "basarili"

            if (x2 - x1) >= MIN_FACE_SIZE and (y2 - y1) >= MIN_FACE_SIZE:
                pose, _ = estimate_pose_label(frame)
                panel_data["landmark_analysis"] = "tamamlandi"
                panel_data["pose"] = pose

                blur_value = measure_blur(face_crop)
                bright_value = measure_brightness(face_crop)
                panel_data["blur_score"] = round(blur_value, 2)
                panel_data["brightness"] = round(bright_value, 2)

                if pose in ("right", "left"):
                    strict_threshold = LBPH_STRICT_THRESHOLD + 5
                    soft_threshold = LBPH_SOFT_THRESHOLD + 5
                    min_required_repeats = MIN_REPEATS_PROFILE
                elif pose in ("up", "down"):
                    strict_threshold = LBPH_STRICT_THRESHOLD + 3
                    soft_threshold = LBPH_SOFT_THRESHOLD + 3
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
                        frame,
                        "unknown | dusuk isik",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                    last_output = build_output(
                        user_id="N/A",
                        name="unknown",
                        recognized=False,
                        confidence=0,
                        pose=pose,
                        next_step="goruntu cok karanlik, kabul edilmedi"
                    )

                elif blur_value < BLUR_THRESHOLD:
                    panel_data["next_step"] = "goruntu bulanik, tekrar deneyin"

                else:
                    if frame_counter % PREDICT_EVERY_N_FRAMES == 0:
                        processed = preprocess_face(face_crop)
                        label, conf = recognizer.predict(processed)
                        display_score = conf_to_display_score(conf)
                        user = find_user_by_numeric_id(label) if label is not None else None

                        if conf <= strict_threshold and user is not None:
                            recent_names.append(user["name"])
                            recent_ids.append(user["user_id"])
                            recent_scores.append(display_score)
                            recent_valid.append(True)

                        elif conf <= soft_threshold and user is not None:
                            recent_names.append(user["name"])
                            recent_ids.append(user["user_id"])
                            recent_scores.append(display_score)
                            recent_valid.append("soft")

                        else:
                            recent_names.append("unknown")
                            recent_ids.append("N/A")
                            recent_scores.append(display_score)
                            recent_valid.append(False)

                    if recent_scores:
                        avg_score = round(sum(recent_scores) / len(recent_scores), 2)
                    else:
                        avg_score = 0

                    stable_name, top_count, second_count = get_top_two_counts(recent_names)
                    stable_id = most_common_value(recent_ids)

                    accept = False

                    if (
                        stable_name not in [None, "unknown"]
                        and stable_id not in [None, "N/A"]
                        and top_count >= min_required_repeats
                        and (top_count - second_count) >= DOMINANCE_DIFF
                        and avg_score >= MIN_DISPLAY_SCORE_TO_ACCEPT
                    ):
                        accept = True
                    # Sadece iki aday arasındaki fark devasaysa kabul et
                    if (top_count - second_count) < 20:  # Fark 20'den azsa asla isim söyleme
                        accept = False
                        panel_data["next_step"] = "Kimlik netlestirilemiyor..."

                    if accept:
                        panel_data["recognized_user"] = stable_name
                        panel_data["user_id"] = stable_id
                        panel_data["confidence_score"] = avg_score
                        panel_data["next_step"] = "bellek kontrolune gonderildi"

                        cv2.putText(
                            frame,
                            f"{stable_name} | {stable_id} | %{avg_score:.1f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 255, 0),
                            2
                        )

                        last_output = build_output(
                            user_id=stable_id,
                            name=stable_name,
                            recognized=True,
                            confidence=avg_score,
                            pose=pose,
                            next_step="bellek kontrolune gonderildi"
                        )
                    else:
                        panel_data["recognized_user"] = "unknown"
                        panel_data["user_id"] = "N/A"
                        panel_data["confidence_score"] = avg_score
                        panel_data["next_step"] = "kararsiz, bellek kontrolune gonderilmedi"

                        cv2.putText(
                            frame,
                            f"unknown | %{avg_score:.1f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 0, 255),
                            2
                        )

                        last_output = build_output(
                            user_id="N/A",
                            name="unknown",
                            recognized=False,
                            confidence=avg_score,
                            pose=pose,
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


    cap.release()
    cv2.destroyAllWindows()