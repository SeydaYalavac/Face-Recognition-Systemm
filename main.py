# ============================================================================
# YÜZ TANIMA SİSTEMİ - ANA PROGRAM
# ============================================================================
# Tüm modüller import edilerek burada koordine edilir
# ============================================================================
from face_utils import ensure_paths, load_users
from train_model import train_lbph_model
from clear_user_folder import clear_user_folder
from bellek_utils import BellekSistemi
from kisi4_interface import Kisi4Interface
from access_control import donanim_tetikle, log_kaydet, erisim_karari_uret, log_goruntule


def ana_menu():
    """Ana menüyü göster"""
    print("\n" + "="*60)
    print("     YÜZ TANIMA SİSTEMİ - BÜTÜNLEŞTIRILMIŞ VERSIYON")
    print("="*60)
    print("\n1 - Yeni kullanici kaydet")
    print("2 - Modeli egit")
    print("3 - Canli tanima baslat")
    print("4 - Fotoğraf klasörünü temizle")
    print("5 - Bellek haritasini goster")
    print("6 - Erisim kayitlarini goster")
    print("7 - Cikis")
    print("="*60)


def main():
    """Ana program - Tüm modülleri koordine et"""
    
    # Başlatma
    ensure_paths()
    bellek = BellekSistemi()
    interface = Kisi4Interface(bellek)
    
    print("\n✓ Sistem başlatıldı!")
    print(f"✓ Bellek kapasite: {bellek.bellek_istatistikleri()['total_capacity']} kullanıcı")
    print(f"✓ Aktif kullanıcılar: {bellek.bellek_istatistikleri()['used']}")
    
    while True:
        ana_menu()
        secim = input("Seciminiz: ").strip()

        if secim == "1":
            name = input("Kullanici adi: ").strip()
            if name:
                print(f"Yeni kullanici kaydi başlatılıyor: {name}")

                # --- BURAYI DÜZELTİYORUZ ---
                from register_user import register_new_user  # Kayıt modülünü çağırıyoruz
                register_new_user(name)  # Fonksiyonu çalıştırıyoruz
                # ---------------------------

            else:
                print("Gecersiz isim.")

        elif secim == "2":
            print("\nModel eğitimi başlatılıyor...")
            train_lbph_model()


        elif secim == "3":

            print("\nCanlı tanıma başlatılıyor...")

            print("(Kamera açılacak - Q tuşu ile çıkın)")

            # BURAYI DÜZELT: Sadece yazı vardı, şimdi fonksiyonu çağırıyoruz

            from recognize_user import recognize_live

            recognize_live(bellek)

        elif secim == "4":
            print("\nYüz fotoğraflarını silme işlemi...")
            print("(Klasörler: yuz1, yuz2, yuz3, vb.)")
            folder = input("Hangi klasörü temizlemek istiyorsun? (Örn: yuz3): ").strip()
            if folder:
                onay = input(f"'{folder}' klasöründeki TÜM fotoğrafları silmek istediğinden emin misin? (evet/hayir): ").strip().lower()
                if onay == "evet":
                    clear_user_folder(folder)
                else:
                    print("Iptal edildi.")
            else:
                print("Gecersiz klasor adi.")

        elif secim == "5":
            bellek.bellek_haritasi_goster()

        elif secim == "6":
            bellek.log_goster()

        elif secim == "7":
            print("\nCikis yapildi. Hoşça kalın!")
            break

        else:
            print("Gecersiz secim. Lütfen tekrar deneyin.")


if __name__ == "__main__":
    main()
