from config import *  # Tüm config ayarları
from face_utils import ensure_paths, load_users, register_new_user, train_lbph_model
from face_utils import recognize_live, clear_temp_user_photos, clear_single_user_photos
from face_utils import clear_user_folder
from output_formatter import build_output
from bellek_utils import BellekSistemi
from kisi4_interface import Kisi4Interface
from access_control import donanim_tetikle, log_kaydet, erisim_karari_uret

# Ana fonksiyonlar burada çağrılır
def main():
    ensure_paths()  # face_utils'den
    bellek = BellekSistemi()  # bellek_utils'den
    interface = Kisi4Interface(bellek)  # kisi4_interface'den

    while True:
        print("\n1 - Yeni kullanici kaydet")
        print("2 - Model egit")
        print("3 - Tani")
        print("4 - Cikis")
        choice = input("Secim: ")
        if choice == "1":
            name = input("Isim: ")
            register_new_user(name)  # register_user'den (face_utils'e taşındı)
        elif choice == "2":
            train_lbph_model()  # train_model'den
        elif choice == "3":
            recognize_live()  # recognize_user'den
        elif choice == "4":
            break

if __name__ == "__main__":
    main()