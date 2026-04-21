# main.py içeriği
from config import BASE_ADDRESS  # Ayarları al
from face_utils import recognize_face  # Kişi 1'den fonksiyon çağır
from bellek_utils import get_user_authority  # Kişi 2'den fonksiyon çağır
from access_control import check_access  # Kişi 3'den fonksiyon çağır