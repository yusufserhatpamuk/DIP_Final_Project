import cv2
import numpy as np

class AdvancedInpaintingEngine:
    def __init__(self, alpha=[1/3, 1/3, 1/3], tensor_sigma=1.0, ksize=5):
        self.alpha = alpha
        self.tensor_sigma = tensor_sigma
        self.ksize = ksize

    def compute_multiscale_edges(self, gray_img):
        blur1 = cv2.GaussianBlur(gray_img, (3, 3), 1.0)
        e1 = cv2.Canny(blur1, 50, 150).astype(np.float32) / 255.0
        
        blur2 = cv2.GaussianBlur(gray_img, (5, 5), 2.0)
        e2 = cv2.Canny(blur2, 30, 100).astype(np.float32) / 255.0
        
        blur3 = cv2.GaussianBlur(gray_img, (7, 7), 4.0)
        e3 = cv2.Canny(blur3, 15, 50).astype(np.float32) / 255.0
        
        E = self.alpha[0] * e1 + self.alpha[1] * e2 + self.alpha[2] * e3
        return E

    def compute_structure_tensor_orientation(self, gray_img):
        Ix = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
        
        Ix2 = Ix * Ix
        Iy2 = Iy * Iy
        IxIy = Ix * Iy
        
        Ix2_blur = cv2.GaussianBlur(Ix2, (self.ksize, self.ksize), self.tensor_sigma)
        Iy2_blur = cv2.GaussianBlur(Iy2, (self.ksize, self.ksize), self.tensor_sigma)
        IxIy_blur = cv2.GaussianBlur(IxIy, (self.ksize, self.ksize), self.tensor_sigma)
        
        theta = 0.5 * np.arctan2(2 * IxIy_blur, Ix2_blur - Iy2_blur)
        return theta

    def inpaint(self, image, mask, radius=3, lambda_e=2.0, sigma_I=20.0, use_tensor=True, use_edges=True):
        """
        Gelişmiş Katman Tabanlı Hızlı Anisotropic İn boyama Fonksiyonu.
        Sonsuz döngü riskini tamamen ortadan kaldırır.
        """
        h, w, c = image.shape
        result = image.copy().astype(np.float32)
        
        if len(image.shape) == 3 and c == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        if use_edges:
            E = self.compute_multiscale_edges(gray)
        else:
            E = np.zeros_like(gray, dtype=np.float32)
            
        if use_tensor:
            theta = self.compute_structure_tensor_orientation(gray)
        else:
            theta = None
        
        working_mask = mask.copy()
        
        # Grid indeks koordinatları
        y_indices, x_indices = np.indices((h, w))
        
        # Güvenli döngü için maksimum iterasyon sınırı (Sonsuz döngü koruması)
        max_iters = 500
        iter_count = 0
        
        while np.sum(working_mask > 0) > 0 and iter_count < max_iters:
            iter_count += 1
            
            # Maskenin tam dış çeperindeki sınır piksellerini tespit et
            dilated_mask = cv2.dilate(working_mask, np.ones((3, 3), np.uint8))
            border_mask = (dilated_mask > 0) & (working_mask == 0)
            
            # Doldurulacak olan iç çeper pikselleri
            target_mask = (working_mask > 0) & (cv2.dilate(border_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
            target_pixels = np.argwhere(target_mask)
            
            if len(target_pixels) == 0:
                # Koruma: Eğer sınır saptanamazsa kalan tüm pikselleri zorla doldur
                target_pixels = np.argwhere(working_mask > 0)
            
            next_mask = working_mask.copy()
            
            for p_y, p_x in target_pixels:
                y_min, y_max = max(0, p_y - radius), min(h, p_y + radius + 1)
                x_min, x_max = max(0, p_x - radius), min(w, p_x + radius + 1)
                
                local_mask = working_mask[y_min:y_max, x_min:x_max]
                known_mask = (local_mask == 0)
                
                if not np.any(known_mask):
                    continue
                    
                local_y = y_indices[y_min:y_max, x_min:x_max][known_mask]
                local_x = x_indices[y_min:y_max, x_min:x_max][known_mask]
                local_pixels = result[y_min:y_max, x_min:x_max][known_mask]
                
                # 1. Uzamsal Mesafe
                d_pq = np.sqrt((p_x - local_x)**2 + (p_y - local_y)**2)
                d_pq[d_pq == 0] = 1.0
                
                # 2. Structure Tensor Hizalanması
                if use_tensor and theta is not None:
                    phi_pq = np.arctan2(local_y - p_y, local_x - p_x)
                    # DÜZELTME: Gradyan yönü olan theta'ya dik (yani kenar boyunca) yayılım yapmak için sinüs kullanıyoruz
                    dir_alignment = np.abs(np.sin(theta[p_y, p_x] - phi_pq))
                else:
                    dir_alignment = 1.0
                
                w_pq = (1.0 / d_pq) * dir_alignment
                
                # 3. Multi-Scale Kenar Katkısı
                w_final = w_pq * (1.0 + lambda_e * E[local_y, local_x])
                total_w = np.sum(w_final)
                
                if total_w > 0:
                    result[p_y, p_x] = np.sum(w_final[:, np.newaxis] * local_pixels, axis=0) / total_w
                    next_mask[p_y, p_x] = 0
            
            # Eğer bu iterasyonda hiç piksel silinemediyse döngüyü kır (Kilitlenmeyi önler)
            if np.array_equal(working_mask, next_mask):
                break
                
            working_mask = next_mask
            
        return np.clip(result, 0, 255).astype(np.uint8)