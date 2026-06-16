import os
import argparse
import time
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

from src.inpaint_engine import AdvancedInpaintingEngine
from src.dataset import AdvancedScratchGenerator

def parse_args():
    parser = argparse.ArgumentParser(
        description="Edge-Aware Multi-Scale Structure Tensor Guided Image Inpainting CLI Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", required=True, help="Path to the degraded image or folder containing images.")
    parser.add_argument("--mask", help="Path to the binary scratch mask image. If not provided, a scratch mask will be synthetically generated.")
    parser.add_argument("--reference", help="Path to the clean ground-truth image for PSNR/SSIM evaluation.")
    parser.add_argument("--output_dir", default="./outputs", help="Directory where restored output images will be saved.")
    
    # Algoritma parametreleri
    parser.add_argument("--baseline", choices=["proposed", "telea", "ns", "patch"], default="proposed", help="Select the algorithm: proposed, telea, ns, or patch.")
    parser.add_argument("--radius", type=int, default=3, help="Inpainting neighborhood radius R.")
    parser.add_argument("--lambda_e", type=float, default=2.0, help="Edge-directed constraint enforcement parameter lambda.")
    parser.add_argument("--no_tensor", action="store_true", help="Disable Structure Tensor direction guidance (Ablation mode).")
    parser.add_argument("--no_edges", action="store_true", help="Disable Multi-Scale Edge confidence weighting (Ablation mode).")
    
    # Maske üretim parametreleri (maske verilmediğinde kullanılır)
    parser.add_argument("--density", type=float, default=0.03, help="Synthetic scratch density (ratio of image area).")
    parser.add_argument("--thickness", type=int, default=3, help="Maximum thickness of synthetic scratches in pixels.")
    
    return parser.parse_args()

def process_single_image(img_path, mask_path, ref_path, output_dir, args, engine):
    print(f"\n[PROCESSING] Görüntü: {os.path.basename(img_path)}")
    
    # Degraded ve maskeyi yükle
    degraded_img = cv2.imread(img_path)
    if degraded_img is None:
        print(f"[ERROR] Görüntü yüklenemedi: {img_path}")
        return
        
    h, w, c = degraded_img.shape
    
    # Maskeyi edin
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[ERROR] Maske yüklenemedi: {mask_path}")
            return
        # Maske boyutunun görüntü ile uyuşmasını sağla
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        print(f"[INFO] Maske yolu belirtilmedi. Yoğunluk={args.density*100}%, Kalınlık={args.thickness}px olacak şekilde sentetik maske üretiliyor...")
        sg = AdvancedScratchGenerator(seed=int(time.time()))
        mask = sg.generate_scratch_mask(degraded_img.shape, density=args.density, max_thickness=args.thickness)
        
        # Eğer orijinal referans resmi verilmişse, çizikli resmi oluştur
        if ref_path:
            ref_img = cv2.imread(ref_path)
            if ref_img is not None:
                degraded_img = ref_img.copy()
                degraded_img[mask == 255] = [255, 255, 255]
                print("[INFO] Sentetik çizikler orijinal referans görüntüsüne uygulandı.")
    
    # Inpaint işlemini çalıştır
    t0 = time.time()
    if args.baseline == "telea":
        from src.baselines import apply_telea_inpainting
        print("[INFO] Running Telea Fast Marching Baseline...")
        restored_img = apply_telea_inpainting(degraded_img, mask, radius=args.radius)
    elif args.baseline == "ns":
        from src.baselines import apply_ns_inpainting
        print("[INFO] Running Navier-Stokes (PDE) Baseline...")
        restored_img = apply_ns_inpainting(degraded_img, mask, radius=args.radius)
    elif args.baseline == "patch":
        from src.baselines import apply_patch_inpainting
        print("[INFO] Running Localized Patch-Based Baseline...")
        restored_img = apply_patch_inpainting(degraded_img, mask, patch_size=7, search_window=31)
    else:
        use_tensor = not args.no_tensor
        use_edges = not args.no_edges
        print("[INFO] Running Proposed Edge-Aware Anisotropic Engine...")
        restored_img = engine.inpaint(
            degraded_img, mask, 
            radius=args.radius, 
            lambda_e=args.lambda_e, 
            use_tensor=use_tensor, 
            use_edges=use_edges
        )
    elapsed_time = time.time() - t0
    
    print(f"[SUCCESS] Inpainting tamamlandı. Süre: {elapsed_time:.4f} saniye.")
    
    # Çıktıyı kaydet
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out_name = f"{base_name}_restored.png"
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, restored_img)
    print(f"[SAVED] Sonuç kaydedildi: {out_path}")
    
    # Maskeyi de kaydet (kolay incelemek için)
    mask_out_path = os.path.join(output_dir, f"{base_name}_mask.png")
    cv2.imwrite(mask_out_path, mask)
    
    # Değerlendirme (Referans resim varsa)
    if ref_path:
        ref_img = cv2.imread(ref_path)
        if ref_img is not None:
            if ref_img.shape != restored_img.shape:
                ref_img = cv2.resize(ref_img, (restored_img.shape[1], restored_img.shape[0]))
            
            psnr_val = psnr_metric(ref_img, restored_img)
            ssim_val = ssim_metric(ref_img, restored_img, channel_axis=2)
            print(f"[EVALUATION] PSNR: {psnr_val:.4f} dB | SSIM: {ssim_val:.6f}")

def main():
    args = parse_args()
    engine = AdvancedInpaintingEngine()
    
    # Girişin klasör mü dosya mı olduğunu kontrol et
    if os.path.isdir(args.input):
        print(f"[BATCH MODE] '{args.input}' klasöründeki tüm görüntüler işleniyor...")
        valid_exts = ['.png', '.jpg', '.jpeg', '.bmp']
        files = [os.path.join(args.input, f) for f in os.listdir(args.input) 
                 if os.path.splitext(f)[1].lower() in valid_exts]
        
        if not files:
            print("[WARNING] Klasörde geçerli görüntü bulunamadı.")
            return
            
        for f in files:
            process_single_image(f, args.mask, args.reference, args.output_dir, args, engine)
    else:
        process_single_image(args.input, args.mask, args.reference, args.output_dir, args, engine)

if __name__ == "__main__":
    main()
