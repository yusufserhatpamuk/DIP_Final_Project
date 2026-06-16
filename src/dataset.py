import os
import cv2
import numpy as np
import requests

class KodakDatasetManager:
    """Kodak24 veri setini yöneten ve otomatik indiren sınıf."""
    def __init__(self, data_dir="./data/kodak"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def download_dataset(self):
        print("[INFO] Kodak24 veri seti kontrol ediliyor...")
        # Kodak24 URL şablonu (True Color Image Suite)
        base_url = "http://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"
        
        for i in range(1, 25):
            file_name = f"kodim{i:02d}.png"
            file_path = os.path.join(self.data_dir, file_name)
            
            if not os.path.exists(file_path):
                url = base_url.format(i)
                print(f"[DOWNLOADING] {file_name} indiriliyor...")
                response = requests.get(url)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                else:
                    print(f"[ERROR] {file_name} indirilemedi!")
        print("[INFO] Veri seti hazır.")

class AdvancedScratchGenerator:
    """Gerçekçi ve rastgele çizik/çatlak maskesi üreten gelişmiş jeneratör."""
    def __init__(self, seed=42):
        np.random.seed(seed)
        
    def generate_scratch_mask(self, image_shape, density=0.02, max_thickness=3):
        """
        Belirlenen yoğunluk (density) ve kalınlığa göre rastgele eğriler (Bézier)
        kullanarak çizik maskesi üretir.
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Toplam piksel sayısına göre çizik uzunluğunu belirle
        total_pixels = h * w
        target_scratch_pixels = total_pixels * density
        current_scratch_pixels = 0
        
        while current_scratch_pixels < target_scratch_pixels:
            # Rastgele başlangıç, kontrol ve bitiş noktaları (Bézier Eğrisi için)
            start_p = (np.random.randint(0, w), np.random.randint(0, h))
            control_p = (np.random.randint(0, w), np.random.randint(0, h))
            end_p = (np.random.randint(0, w), np.random.randint(0, h))
            
            thickness = np.random.randint(1, max_thickness + 1)
            
            # Eğriyi piksellere dökme (t: 0->1 arası adımlar)
            points = []
            for t in np.linspace(0, 1, 100):
                # Quadratic Bézier formülü
                px = int((1-t)**2 * start_p[0] + 2*(1-t)*t * control_p[0] + t**2 * end_p[0])
                py = int((1-t)**2 * start_p[1] + 2*(1-t)*t * control_p[1] + t**2 * end_p[1])
                if 0 <= px < w and 0 <= py < h:
                    points.append((px, py))
            
            # Eğriyi maske üzerine çiz
            for i in range(len(points) - 1):
                cv2.line(mask, points[i], points[i+1], 255, thickness)
                
            current_scratch_pixels = np.sum(mask == 255)
            
        return mask