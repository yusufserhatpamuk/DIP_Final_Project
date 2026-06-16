import os
import time
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

from src.dataset import KodakDatasetManager, AdvancedScratchGenerator
from src.baselines import apply_telea_inpainting, apply_ns_inpainting, apply_patch_inpainting
from src.inpaint_engine import AdvancedInpaintingEngine

def run_benchmarks():
    dm = KodakDatasetManager()
    dm.download_dataset()
    sg = AdvancedScratchGenerator(seed=42)
    engine = AdvancedInpaintingEngine()
    
    os.makedirs("./results", exist_ok=True)
    
    densities = [0.01, 0.03, 0.05] 
    thicknesses = [2, 4]           
    
    benchmark_data = []
    images_to_test = [1, 2, 3]
    
    print("\n[START] Deney matrisi koşturuluyor. Bu işlem biraz zaman alabilir...")
    
    for density in densities:
        for thickness in thicknesses:
            print(f"\n--- Senaryo: Yoğunluk=%{density*100}, Kalınlık={thickness}px ---")
            
            for img_idx in images_to_test:
                img_name = f"kodim{img_idx:02d}.png"
                img_path = os.path.join(dm.data_dir, img_name)
                
                orig_img = cv2.imread(img_path)
                if orig_img is None: continue
                
                mask = sg.generate_scratch_mask(orig_img.shape, density=density, max_thickness=thickness)
                degraded_img = orig_img.copy()
                degraded_img[mask == 255] = [255, 255, 255] 
                print(f"   [PROCESSING] {img_name} görüntüsü inpaint ediliyor...")
                
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
                res_patch = apply_patch_inpainting(degraded_img, mask, patch_size=7, search_window=31)
                time_patch = time.time() - t0
                
                # --- PROPOSED METHOD ---
                t0 = time.time()
                res_proposed = engine.inpaint(degraded_img, mask, radius=5, lambda_e=2.0)
                time_proposed = time.time() - t0
                
                psnr_t = psnr_metric(orig_img, res_telea)
                ssim_t = ssim_metric(orig_img, res_telea, channel_axis=2)
                
                psnr_p = psnr_metric(orig_img, res_pde)
                ssim_p = ssim_metric(orig_img, res_pde, channel_axis=2)
                
                psnr_patch = psnr_metric(orig_img, res_patch)
                ssim_patch = ssim_metric(orig_img, res_patch, channel_axis=2)
                
                psnr_prop = psnr_metric(orig_img, res_proposed)
                ssim_prop = ssim_metric(orig_img, res_proposed, channel_axis=2)
                
                benchmark_data.append({
                    "Image": img_name, "Density": density, "Thickness": thickness,
                    "Telea_PSNR": psnr_t, "Telea_SSIM": ssim_t, "Telea_Time": time_telea,
                    "PDE_PSNR": psnr_p, "PDE_SSIM": ssim_p, "PDE_Time": time_pde,
                    "Patch_PSNR": psnr_patch, "Patch_SSIM": ssim_patch, "Patch_Time": time_patch,
                    "Proposed_PSNR": psnr_prop, "Proposed_SSIM": ssim_prop, "Proposed_Time": time_proposed
                })
                
                if img_idx == 1:
                    fig, axes = plt.subplots(1, 6, figsize=(24, 5))
                    axes[0].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)); axes[0].set_title("Orijinal")
                    axes[1].imshow(cv2.cvtColor(degraded_img, cv2.COLOR_BGR2RGB)); axes[1].set_title("Çizikli Görüntü")
                    axes[2].imshow(cv2.cvtColor(res_telea, cv2.COLOR_BGR2RGB)); axes[2].set_title(f"Telea\nPSNR: {psnr_t:.2f}")
                    axes[3].imshow(cv2.cvtColor(res_pde, cv2.COLOR_BGR2RGB)); axes[3].set_title(f"PDE (NS)\nPSNR: {psnr_p:.2f}")
                    axes[4].imshow(cv2.cvtColor(res_patch, cv2.COLOR_BGR2RGB)); axes[4].set_title(f"Patch-Based\nPSNR: {psnr_patch:.2f}")
                    axes[5].imshow(cv2.cvtColor(res_proposed, cv2.COLOR_BGR2RGB)); axes[5].set_title(f"Önerilen\nPSNR: {psnr_prop:.2f}")
                    for ax in axes: ax.axis('off')
                    plt.savefig(f"./results/comparison_d{density}_t{thickness}.png", bbox_inches='tight')
                    plt.close()

    df = pd.DataFrame(benchmark_data)
    df.to_csv("./results/quantitative_results.csv", index=False)
    
    print("\n================ DENEY SONUÇLARI ÖZETİ (ORTALAMA) ================")
    summary = df.mean(numeric_only=True)
    print(f"Telea Ortalama PSNR: {summary['Telea_PSNR']:.2f} dB, SSIM: {summary['Telea_SSIM']:.4f}, Süre: {summary['Telea_Time']:.4f}s")
    print(f"PDE    Ortalama PSNR: {summary['PDE_PSNR']:.2f} dB, SSIM: {summary['PDE_SSIM']:.4f}, Süre: {summary['PDE_Time']:.4f}s")
    print(f"Patch  Ortalama PSNR: {summary['Patch_PSNR']:.2f} dB, SSIM: {summary['Patch_SSIM']:.4f}, Süre: {summary['Patch_Time']:.4f}s")
    print(f"Önerilen Ortalama PSNR: {summary['Proposed_PSNR']:.2f} dB, SSIM: {summary['Proposed_SSIM']:.4f}, Süre: {summary['Proposed_Time']:.4f}s")
    print("==================================================================")
    print("[SUCCESS] Tüm simülasyonlar tamamlandı. Sonuçlar 'results/' klasörüne kaydedildi.")

if __name__ == "__main__":
    run_benchmarks()