import os
import time
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

from src.dataset import KodakDatasetManager, AdvancedScratchGenerator
from src.baselines import apply_telea_inpainting, apply_ns_inpainting, apply_patch_inpainting, apply_lama_inpainting
from src.inpaint_engine import AdvancedInpaintingEngine

# Görüntü Grupları (Hocanın "class-level analysis" isteği için)
IMAGE_CLASSES = {
    "Landscapes": [1, 2, 13, 14, 16, 21, 23],
    "Portraits/People": [4, 12, 15, 17, 18],
    "Urban/Structures": [3, 5, 8, 9, 22, 24],
    "High-Texture/Detail": [6, 7, 10, 11, 19, 20]
}

def get_image_class(img_idx):
    for class_name, indices in IMAGE_CLASSES.items():
        if img_idx in indices:
            return class_name
    return "Other"

def run_advanced_benchmarks():
    dm = KodakDatasetManager()
    dm.download_dataset()
    sg = AdvancedScratchGenerator(seed=42)
    engine = AdvancedInpaintingEngine()
    
    os.makedirs("./results", exist_ok=True)
    
    # Deney matrisi
    densities = [0.01, 0.03, 0.05]
    thicknesses = [2, 4]
    
    benchmark_data = []
    
    print("\n[START] Gelişmiş deney matrisi koşturuluyor (24 Kodak Görüntüsü x 5 Model)...")
    
    # Tüm 24 resim için
    for img_idx in range(1, 25):
        img_name = f"kodim{img_idx:02d}.png"
        img_path = os.path.join(dm.data_dir, img_name)
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            print(f"[WARNING] {img_name} okunamadı, atlanıyor...")
            continue
            
        img_class = get_image_class(img_idx)
        print(f"\nİşleniyor: {img_name} ({img_class})...")
        
        for density in densities:
            for thickness in thicknesses:
                # Maske üret
                mask = sg.generate_scratch_mask(orig_img.shape, density=density, max_thickness=thickness)
                degraded_img = orig_img.copy()
                degraded_img[mask == 255] = [255, 255, 255]
                
                # --- BASELINE 1: TELEA ---
                t0 = time.time()
                res_telea = apply_telea_inpainting(degraded_img, mask, radius=3)
                time_telea = time.time() - t0
                
                # --- BASELINE 2: PDE (NS) ---
                t0 = time.time()
                res_pde = apply_ns_inpainting(degraded_img, mask, radius=3)
                time_pde = time.time() - t0
                
                # --- BASELINE 3: PATCH-BASED (EXEMPLAR) ---
                t0 = time.time()
                res_patch = apply_patch_inpainting(degraded_img, mask, patch_size=7, search_window=15)
                time_patch = time.time() - t0
                
                # --- BASELINE 4: LAMA DL-BASED ---
                t0 = time.time()
                res_lama = apply_lama_inpainting(degraded_img, mask)
                time_lama = time.time() - t0
                
                # --- PROPOSED METHOD (FULL) ---
                t0 = time.time()
                res_proposed = engine.inpaint(degraded_img, mask, radius=3, lambda_e=2.0, use_tensor=True, use_edges=True)
                time_proposed = time.time() - t0
                
                # --- ABLATION 1: W/O EDGES (use_edges=False) ---
                t0 = time.time()
                res_no_edges = engine.inpaint(degraded_img, mask, radius=3, lambda_e=2.0, use_tensor=True, use_edges=False)
                time_no_edges = time.time() - t0
                
                # --- ABLATION 2: W/O TENSOR (use_tensor=False) ---
                t0 = time.time()
                res_no_tensor = engine.inpaint(degraded_img, mask, radius=3, lambda_e=2.0, use_tensor=False, use_edges=True)
                time_no_tensor = time.time() - t0
                
                # Metrikleri hesapla
                psnr_t = psnr_metric(orig_img, res_telea)
                ssim_t = ssim_metric(orig_img, res_telea, channel_axis=2)
                
                psnr_p = psnr_metric(orig_img, res_pde)
                ssim_p = ssim_metric(orig_img, res_pde, channel_axis=2)
                
                psnr_patch = psnr_metric(orig_img, res_patch)
                ssim_patch = ssim_metric(orig_img, res_patch, channel_axis=2)
                
                psnr_lama = psnr_metric(orig_img, res_lama)
                ssim_lama = ssim_metric(orig_img, res_lama, channel_axis=2)
                
                psnr_prop = psnr_metric(orig_img, res_proposed)
                ssim_prop = ssim_metric(orig_img, res_proposed, channel_axis=2)
                
                psnr_no_e = psnr_metric(orig_img, res_no_edges)
                ssim_no_e = ssim_metric(orig_img, res_no_edges, channel_axis=2)
                
                psnr_no_t = psnr_metric(orig_img, res_no_tensor)
                ssim_no_t = ssim_metric(orig_img, res_no_tensor, channel_axis=2)
                
                benchmark_data.append({
                    "Image": img_name,
                    "Class": img_class,
                    "Density": density,
                    "Thickness": thickness,
                    "Telea_PSNR": psnr_t, "Telea_SSIM": ssim_t, "Telea_Time": time_telea,
                    "PDE_PSNR": psnr_p, "PDE_SSIM": ssim_p, "PDE_Time": time_pde,
                    "Patch_PSNR": psnr_patch, "Patch_SSIM": ssim_patch, "Patch_Time": time_patch,
                    "LaMa_PSNR": psnr_lama, "LaMa_SSIM": ssim_lama, "LaMa_Time": time_lama,
                    "Proposed_PSNR": psnr_prop, "Proposed_SSIM": ssim_prop, "Proposed_Time": time_proposed,
                    "Ablation_NoEdges_PSNR": psnr_no_e, "Ablation_NoEdges_SSIM": ssim_no_e, "Ablation_NoEdges_Time": time_no_edges,
                    "Ablation_NoTensor_PSNR": psnr_no_t, "Ablation_NoTensor_SSIM": ssim_no_t, "Ablation_NoTensor_Time": time_no_tensor
                })

    df = pd.DataFrame(benchmark_data)
    df.to_csv("./results/advanced_quantitative_results.csv", index=False)
    print("\n[SUCCESS] Tüm simülasyon metrisi tamamlandı. Ham veriler kaydedildi.")
    
    # 1. Genel Ortalama Sonuçları
    summary_mean = df.mean(numeric_only=True)
    print("\n================ GENEL DENEY SONUÇLARI (ORTALAMA) ================")
    print(f"Telea            PSNR: {summary_mean['Telea_PSNR']:.2f} dB, SSIM: {summary_mean['Telea_SSIM']:.4f}, Süre: {summary_mean['Telea_Time']:.4f}s")
    print(f"PDE (NS)         PSNR: {summary_mean['PDE_PSNR']:.2f} dB, SSIM: {summary_mean['PDE_SSIM']:.4f}, Süre: {summary_mean['PDE_Time']:.4f}s")
    print(f"Patch-Based      PSNR: {summary_mean['Patch_PSNR']:.2f} dB, SSIM: {summary_mean['Patch_SSIM']:.4f}, Süre: {summary_mean['Patch_Time']:.4f}s")
    print(f"LaMa (WACV 2022) PSNR: {summary_mean['LaMa_PSNR']:.2f} dB, SSIM: {summary_mean['LaMa_SSIM']:.4f}, Süre: {summary_mean['LaMa_Time']:.4f}s")
    print(f"Önerilen (Full)  PSNR: {summary_mean['Proposed_PSNR']:.2f} dB, SSIM: {summary_mean['Proposed_SSIM']:.4f}, Süre: {summary_mean['Proposed_Time']:.4f}s")
    print(f"Ablasyon (No_Edg)PSNR: {summary_mean['Ablation_NoEdges_PSNR']:.2f} dB, SSIM: {summary_mean['Ablation_NoEdges_SSIM']:.4f}, Süre: {summary_mean['Ablation_NoEdges_Time']:.4f}s")
    print(f"Ablasyon (No_Ten)PSNR: {summary_mean['Ablation_NoTensor_PSNR']:.2f} dB, SSIM: {summary_mean['Ablation_NoTensor_SSIM']:.4f}, Süre: {summary_mean['Ablation_NoTensor_Time']:.4f}s")
    
    # 2. Sınıf Düzeyinde Ortalama Analiz (Class-Level Analysis)
    class_summary = df.groupby("Class").mean(numeric_only=True)
    class_summary.to_csv("./results/class_level_results.csv")
    print("\n================ SINIF DÜZEYİNDE ANALİZ (PSNR dB) ================")
    for class_name in class_summary.index:
        row = class_summary.loc[class_name]
        print(f"Sınıf: {class_name}")
        print(f"  * Telea: {row['Telea_PSNR']:.2f} dB | PDE: {row['PDE_PSNR']:.2f} dB | Patch-Based: {row['Patch_PSNR']:.2f} dB | LaMa: {row['LaMa_PSNR']:.2f} dB | Önerilen: {row['Proposed_PSNR']:.2f} dB")
        print(f"  * Ablasyon (NoEdges): {row['Ablation_NoEdges_PSNR']:.2f} dB | Ablasyon (NoTensor): {row['Ablation_NoTensor_PSNR']:.2f} dB")
        
    # 3. Ablasyon Özeti
    ablation_summary = df[[
        "Proposed_PSNR", "Proposed_SSIM",
        "Ablation_NoEdges_PSNR", "Ablation_NoEdges_SSIM",
        "Ablation_NoTensor_PSNR", "Ablation_NoTensor_SSIM"
    ]].mean()
    ablation_summary.to_csv("./results/ablation_results.csv")

def generate_visualizations():
    print("\n[START] Açıklanabilirlik ve Karşılaştırma Görselleri Üretiliyor...")
    dm = KodakDatasetManager()
    sg = AdvancedScratchGenerator(seed=42)
    engine = AdvancedInpaintingEngine()
    
    # kodim03.png (Urban/Structures sınıfından, kenarlar çok belirgin)
    img_path = os.path.join(dm.data_dir, "kodim03.png")
    orig_img = cv2.imread(img_path)
    h, w, c = orig_img.shape
    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    
    # Ara adımları çek
    blur1 = cv2.GaussianBlur(gray_img, (3, 3), 1.0)
    e1 = cv2.Canny(blur1, 50, 150)
    blur2 = cv2.GaussianBlur(gray_img, (5, 5), 2.0)
    e2 = cv2.Canny(blur2, 30, 100)
    blur3 = cv2.GaussianBlur(gray_img, (7, 7), 4.0)
    e3 = cv2.Canny(blur3, 15, 50)
    
    E = engine.compute_multiscale_edges(gray_img)
    theta = engine.compute_structure_tensor_orientation(gray_img)
    
    # 1. Açıklanabilirlik (Interpretability) Görseli
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes[0, 0].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Orijinal Görüntü", fontsize=14)
    
    axes[0, 1].imshow(e1, cmap='gray')
    axes[0, 1].set_title("Canny Kenar (σ=1.0)", fontsize=14)
    
    axes[0, 2].imshow(e2, cmap='gray')
    axes[0, 2].set_title("Canny Kenar (σ=2.0)", fontsize=14)
    
    axes[0, 3].imshow(e3, cmap='gray')
    axes[0, 3].set_title("Canny Kenar (σ=4.0)", fontsize=14)
    
    # Kenar güven haritası
    im_e = axes[1, 0].imshow(E, cmap='jet')
    axes[1, 0].set_title("Çok Ölçekli Kenar Güven Haritası E(p)", fontsize=14)
    fig.colorbar(im_e, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Yapı tensörü yönelimi
    im_theta = axes[1, 1].imshow(theta, cmap='twilight')
    axes[1, 1].set_title("Yapı Tensörü Açı Haritası θ_p", fontsize=14)
    fig.colorbar(im_theta, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Açı haritası histogramı
    axes[1, 2].hist(theta.ravel(), bins=100, color='purple', alpha=0.7)
    axes[1, 2].set_title("Yönelim Dağılım Histogramı", fontsize=14)
    axes[1, 2].set_xlabel("Açı (Radyan)")
    
    axes[1, 3].axis('off')
    
    for r in range(2):
        for col in range(4):
            if not (r == 1 and col == 3):
                if not (r == 1 and col == 2):
                    axes[r, col].axis('off')
                    
    plt.tight_layout()
    plt.savefig("./results/interpretability_kodim03.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   [SAVED] results/interpretability_kodim03.png")
    
    # 2. Kalitatif Karşılaştırma Görseli (Tüm Modeller - Detaylı Kırpma)
    mask = sg.generate_scratch_mask(orig_img.shape, density=0.03, max_thickness=4)
    degraded_img = orig_img.copy()
    degraded_img[mask == 255] = [255, 255, 255]
    
    res_telea = apply_telea_inpainting(degraded_img, mask, radius=3)
    res_pde = apply_ns_inpainting(degraded_img, mask, radius=3)
    res_patch = apply_patch_inpainting(degraded_img, mask, patch_size=7, search_window=15)
    res_lama = apply_lama_inpainting(degraded_img, mask)
    res_proposed = engine.inpaint(degraded_img, mask, radius=3, lambda_e=2.0, use_tensor=True, use_edges=True)
    res_no_tensor = engine.inpaint(degraded_img, mask, radius=3, lambda_e=2.0, use_tensor=False, use_edges=True)
    res_no_edges = engine.inpaint(degraded_img, mask, radius=3, lambda_e=2.0, use_tensor=True, use_edges=False)
    
    # Belirgin bir çizik kesişim bölgesini kırp (kodim03 için kapı kenarı / anahtar deliği civarı)
    # kodim03 boyutu 768 (yükseklik) x 512 (genişlik)
    cy, cx = int(h * 0.45), int(w * 0.35)
    size = 120
    crop_orig = orig_img[cy:cy+size, cx:cx+size]
    crop_degraded = degraded_img[cy:cy+size, cx:cx+size]
    crop_telea = res_telea[cy:cy+size, cx:cx+size]
    crop_pde = res_pde[cy:cy+size, cx:cx+size]
    crop_patch = res_patch[cy:cy+size, cx:cx+size]
    crop_lama = res_lama[cy:cy+size, cx:cx+size]
    crop_prop = res_proposed[cy:cy+size, cx:cx+size]
    crop_no_ten = res_no_tensor[cy:cy+size, cx:cx+size]
    crop_no_edg = res_no_edges[cy:cy+size, cx:cx+size]
    
    titles = [
        "Orijinal Referans", "Çizikli Giriş", 
        "Telea Baseline", "PDE (NS) Baseline", "Patch-Based Baseline",
        "LaMa DL Baseline", "Önerilen (Full)", "Önerilen w/o Tensör", "Önerilen w/o Kenar"
    ]
    crops = [
        crop_orig, crop_degraded, 
        crop_telea, crop_pde, crop_patch,
        crop_lama,
        crop_prop, crop_no_ten, crop_no_edg
    ]
    
    fig, axes = plt.subplots(1, 9, figsize=(31.5, 4))
    for idx, (crop, title) in enumerate(zip(crops, titles)):
        axes[idx].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(title, fontsize=11, pad=10)
        axes[idx].axis('off')
        
    plt.tight_layout()
    plt.savefig("./results/comparison_all_methods_kodim03.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("   [SAVED] results/comparison_all_methods_kodim03.png")

if __name__ == "__main__":
    run_advanced_benchmarks()
    generate_visualizations()
    print("\n[SUCCESS] Tüm deneysel veriler ve görseller başarıyla tamamlandı!")
