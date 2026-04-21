import os
import glob
from face_utils import ensure_paths, load_users
from register_user import register_new_user
from train_model import train_lbph_model
from recognize_user import recognize_live
from config import YUZZ_DIR


def clear_temp_user_photos():
    """
    numeric_id >= 3 olan kullanicilarin fotograflarini siler.
    Seyda (ID 1) ve Sevde (ID 2) korunur.
    Klasorler ve users.json kayitlari silinmez.
    """
    users = load_users()

    if not users:
        print("Kayitli kullanici bulunamadi.")
        return

    silinen_toplam = 0
    islenen_kullanici = 0

    for user in users:
        if user["numeric_id"] <= 2:
            print(f"Korundu: {user['name']} ({user['user_id']})")
            continue

        folder_path = os.path.join(YUZZ_DIR, user["folder"])

        if not os.path.exists(folder_path):
            print(f"Klasor bulunamadi: {folder_path}")
            continue

        files = (
            glob.glob(os.path.join(folder_path, "*.jpg")) +
            glob.glob(os.path.join(folder_path, "*.jpeg")) +
            glob.glob(os.path.join(folder_path, "*.png"))
        )

        if not files:
            print(f"{user['name']} icin silinecek fotograf yok.")
            continue

        for f in files:
            os.remove(f)

        silinen_toplam += len(files)
        islenen_kullanici += 1

        print(f"Silindi: {user['name']} ({user['user_id']}) - {len(files)} fotograf")

    print(f"\nToplam {islenen_kullanici} kullanicinin {silinen_toplam} fotografu silindi.")
    print("Seyda ve Sevde korundu.")


def clear_single_user_photos(user_name: str):
    """
    Belirtilen kullanicinin fotograflarini siler.
    Eger kullanici Seyda veya Sevde ise islemi reddeder.
    """
    users = load_users()
    target_user = None

    for user in users:
        if user["name"].strip().lower() == user_name.strip().lower():
            target_user = user
            break

    if target_user is None:
        print(f"Kullanici bulunamadi: {user_name}")
        return

    if target_user["numeric_id"] <= 2:
        print(f"UYARI: {target_user['name']} korunan kullanicidir. Fotograflari silinemez.")
        return

    folder_path = os.path.join(YUZZ_DIR, target_user["folder"])

    if not os.path.exists(folder_path):
        print(f"Klasor bulunamadi: {folder_path}")
        return

    files = (
        glob.glob(os.path.join(folder_path, "*.jpg")) +
        glob.glob(os.path.join(folder_path, "*.jpeg")) +
        glob.glob(os.path.join(folder_path, "*.png"))
    )

    if not files:
        print(f"{user_name} icin silinecek fotograf bulunamadi.")
        return

    for f in files:
        os.remove(f)

    print(f"{user_name} icin {len(files)} fotograf silindi.")
    print(f"Klasor korundu: {folder_path}")


def main():
    ensure_paths()

    while True:
        print("\n===== KISI 1 - YUZ TANIMA MODULU =====")
        print("1 - Yeni kullanici kaydet")
        print("2 - Modeli egit")
        print("3 - Canli tanima baslat")
        print("4 - Yeni kullanicilarin fotograflarini sil (yuz3+)")
        print("5 - Tek kullanicinin fotograflarini sil")
        print("6 - Cikis")

        secim = input("Seciminiz: ").strip()

        if secim == "1":
            name = input("Kullanici adi: ").strip()
            if name:
                register_new_user(name)
            else:
                print("Gecersiz isim.")

        elif secim == "2":
            train_lbph_model()

        elif secim == "3":
            recognize_live()

        elif secim == "4":
            print("\nSeyda ve Sevde korunacak.")
            print("Diger tum kullanicilarin fotograflari silinecek.")
            onay = input("Emin misiniz? (evet/hayir): ").strip().lower()
            if onay == "evet":
                clear_temp_user_photos()
            else:
                print("Iptal edildi.")

        elif secim == "5":
            name = input("Fotograflari silinecek kullanici adi: ").strip()
            if name:
               2
               clear_single_user_photos(name)
            else:
                print("Gecersiz isim.")

        elif secim == "6":
            print("Cikis yapildi.")
            break

        else:
            print("Gecersiz secim.")


if __name__ == "__main__":
    main()