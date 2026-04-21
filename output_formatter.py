from datetime import datetime

def build_output(user_id, name, recognized, confidence, pose, next_step):
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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