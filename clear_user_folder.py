import os
import shutil
from config import YUZZ_DIR


def clear_user_folder(folder_name="yuz3"):
    folder_path = os.path.join(YUZZ_DIR, folder_name)

    if not os.path.exists(folder_path):
        print(f"Klasör bulunamadı: {folder_path}")
        return False

    file_count = 0
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                file_count += 1
        except Exception as e:
            print(f"Hata: {file} silinemedi → {e}")

    print(f"{folder_name} klasöründeki {file_count} fotoğraf silindi.")
    print(f"Şimdi '1 - Yeni kullanici kaydet' ile tekrar {folder_name} klasörüne kayıt alabilirsiniz.")
    return True


if __name__ == "__main__":
    print("=== Sadece Belirli Kullanıcının Fotoğraflarını Temizleme Aracı ===\n")
    print("Not: yuz1 ve yuz2 korunacak, sadece istediğin yuzX temizlenecek.\n")

    folder = input("Hangi klasörü temizlemek istiyorsun? (Örn: yuz3): ").strip()

    if not folder.startswith("yuz"):
        print("Geçersiz klasör adı. Örnek: yuz3, yuz4, yuz5 ...")
    else:
        confirm = input(f"\n{folder} klasöründeki TÜM fotoğrafları silmek istediğinden emin misin? (E/H): ").strip().lower()

        if confirm in ["e", "evet", "y"]:
            clear_user_folder(folder)
        else:
            print("İşlem iptal edildi.")