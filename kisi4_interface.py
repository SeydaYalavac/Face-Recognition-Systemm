# kisi4_interface.py
"""
Kişi 4'e ham veri sağlama interface'i
SADECE VERİ, GÖRÜNTÜ YOK!
"""

from bellek_utils import BellekSistemi


class Kisi4Interface:
    """Kişi 4'ün kullanacağı interface"""
    
    def __init__(self, bellek):
        self.bellek = bellek
    
    def get_memory_map(self):
        """
        Bellek haritası verilerini ver
        
        Returns:
            dict: Ham veri (formatlama yok!)
        """
        users = []
        for user in self.bellek.tum_kullanicilar():
            users.append({
                "base_address": user["base_address"],
                "user_id": user["user_id"],
                "authority_level": user["authority_level"],
                "status": user["status"],
                "failed_attempts": user["failed_attempts"],
                "alarm": user["alarm"],
                "last_access": user["last_access"]
            })
        
        return {
            "statistics": self.bellek.bellek_istatistikleri(),
            "users": users
        }
    
    def get_dashboard(self):
        """Dashboard için ham veri"""
        stats = self.bellek.bellek_istatistikleri()
        users = self.bellek.tum_kullanicilar()
        
        return {
            "memory_stats": stats,
            "user_count": len(users),
            "active_count": len([u for u in users if u["status"] == 1]),
            "alarm_count": len([u for u in users if u["alarm"] == 1])
        }