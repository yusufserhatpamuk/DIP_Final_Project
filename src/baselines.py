import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def apply_telea_inpainting(image, mask, radius=3):
    """Alexandru Telea (2004) Fast Marching tabanlı inpainting."""
    return cv2.inpaint(image, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)

def apply_ns_inpainting(image, mask, radius=3):
    """Navier-Stokes (PDE) tabanlı inpainting."""
    return cv2.inpaint(image, mask, inpaintRadius=radius, flags=cv2.INPAINT_NS)

def apply_patch_inpainting(image, mask, patch_size=7, search_window=15):
    """
    Hızlı ve lokalize bir yamalı (patch-based/exemplar) inpainting algoritması.
    Inpaint işlemini, maske sınırından içeri doğru (katman katman dıştan içe doldurma) gerçekleştirir.
    Her piksel için, yerel bir arama penceresindeki en benzer bilinen orijinal yamayı bulur.
    """
    h, w, c = image.shape
    result = image.copy()
    working_mask = mask.copy()
    
    # Mesafe transformasyonu kullanarak piksellerin maske sınırına olan uzaklıklarını bul
    dist_transform = cv2.distanceTransform(working_mask, cv2.DIST_L2, 3)
    
    # Maskeli piksellerin koordinatlarını ve uzaklıklarını al
    mask_coords = np.argwhere(working_mask > 0)
    if len(mask_coords) == 0:
        return result
        
    distances = dist_transform[mask_coords[:, 0], mask_coords[:, 1]]
    
    # Uzaklıklara göre sırala (en dış sınır pikselleri ilk sırada)
    sorted_indices = np.argsort(distances)
    sorted_coords = mask_coords[sorted_indices]
    
    half_p = patch_size // 2
    half_s = search_window // 2
    
    # Bilinen pikseller haritası (0: bilinmeyen/maskeli, 1: bilinen/orijinal)
    known_map = np.ones((h, w), dtype=np.uint8)
    known_map[working_mask > 0] = 0
    
    # Orijinal bilinenler haritası (yamaların orijinal temiz piksellerden seçilmesini garanti etmek için)
    known_map_initial = known_map.copy()
    
    # Sınır koordinatlarını yönetmek için dizileri pad'le
    pad = half_p + half_s
    padded_res = np.pad(result, ((pad, pad), (pad, pad), (0, 0)), mode='symmetric')
    padded_known = np.pad(known_map, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
    padded_known_initial = np.pad(known_map_initial, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
    
    for py, px in sorted_coords:
        ppy, ppx = py + pad, px + pad
        
        # Hedef yama ve onun geçerli piksellerinin maskesi
        target_patch = padded_res[ppy - half_p : ppy + half_p + 1, ppx - half_p : ppx + half_p + 1]
        target_known = padded_known[ppy - half_p : ppy + half_p + 1, ppx - half_p : ppx + half_p + 1]
        
        # Arama bölgesinin sınırları
        search_region_res = padded_res[ppy - half_s - half_p : ppy + half_s + half_p + 1, 
                                       ppx - half_s - half_p : ppx + half_s + half_p + 1]
        search_region_known_initial = padded_known_initial[ppy - half_s - half_p : ppy + half_s + half_p + 1, 
                                                           ppx - half_s - half_p : ppx + half_s + half_p + 1]
        
        # Sliding windows üret
        # Görüntü pencereleri boyutu: (S, S, 3, P, P) -> Transpose sonrası: (S, S, P, P, 3)
        img_windows = sliding_window_view(search_region_res, (patch_size, patch_size), axis=(0, 1))
        img_windows = np.transpose(img_windows, (0, 1, 3, 4, 2))
        
        # Başlangıçta bilinen maske pencereleri boyutu: (S, S, P, P)
        known_windows = sliding_window_view(search_region_known_initial, (patch_size, patch_size), axis=(0, 1))
        
        # Sadece tamamen orijinal (bozulmamış) bilinen yamaları aday kabul et
        valid_mask = (np.sum(known_windows, axis=(2, 3)) == patch_size * patch_size)
        valid_indices = np.argwhere(valid_mask)
        
        if len(valid_indices) == 0:
            # Yedek plan: en çok bilinen piksele sahip yamaları seç
            known_sum = np.sum(known_windows, axis=(2, 3))
            max_known = np.max(known_sum)
            if max_known > 0:
                valid_mask = (known_sum == max_known)
                valid_indices = np.argwhere(valid_mask)
                
        if len(valid_indices) == 0:
            continue
            
        # Aday yamaları seç
        valid_patches = img_windows[valid_mask]
        
        # SSD hesapla
        diff = (valid_patches - target_patch[np.newaxis, ...]) * target_known[np.newaxis, ..., np.newaxis]
        ssd = np.sum(diff ** 2, axis=(1, 2, 3))
        
        best_idx = np.argmin(ssd)
        best_coord = valid_indices[best_idx]
        
        # Aday yamanın merkez pikselinin koordinatı
        cy_pad = ppy - half_s + best_coord[0]
        cx_pad = ppx - half_s + best_coord[1]
        
        best_val = padded_res[cy_pad, cx_pad]
        
        # Padded dizilerde ve sonuç dizilerinde doldur
        padded_res[ppy, ppx] = best_val
        padded_known[ppy, ppx] = 1
        result[py, px] = best_val
        
    return result

def apply_lama_inpainting(image, mask):
    """
    LaMa (Resolution-robust Large Mask Inpainting, 2022) deep learning baseline.
    Uses ONNX-wrapped PyTorch model via simple-lama-inpainting.
    """
    from simple_lama_inpainting import SimpleLama
    from PIL import Image

    if not hasattr(apply_lama_inpainting, "_lama_model"):
        apply_lama_inpainting._lama_model = SimpleLama()

    # Convert image to RGB PIL Image
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Convert mask to L PIL Image (255 = region to inpaint)
    pil_mask = Image.fromarray(mask).convert('L')
    
    # Run SimpleLama inpainting
    result_pil = apply_lama_inpainting._lama_model(pil_img, pil_mask)
    
    # Convert back to BGR numpy array
    result_rgb = np.array(result_pil)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    return result_bgr