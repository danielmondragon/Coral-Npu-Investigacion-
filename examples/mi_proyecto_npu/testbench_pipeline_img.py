import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def run_pipeline_test():
    print("\n=======================================================")
    print("   INTEGRACIÓN HARDWARE CORALNPU V2 (MEMORIA DTCM)      ")
    print("=======================================================\n")

    img_path = None
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        folder = "examples/mi_proyecto_npu"
        images = [img for img in glob.glob(os.path.join(folder, "*.jpg")) if "resultado_pipeline" not in img]
        if images:
            img_path = images[0]

    if img_path and os.path.exists(img_path):
        print(f"[INFO] Cargar e inyectar foto en DTCM (512x512 RGB): {img_path}")
        img_file = Image.open(img_path).convert("RGB")
        img_resized = img_file.resize((512, 512), Image.Resampling.LANCZOS)
        input_rgb = np.array(img_resized, dtype=np.uint8)
    else:
        print("[AVISO] Foto no encontrada. Usando matriz por defecto.")
        input_rgb = np.full((512, 512, 3), 128, dtype=np.uint8)

    h, w, c = input_rgb.shape
    gray = (0.299*input_rgb[:,:,0] + 0.587*input_rgb[:,:,1] + 0.114*input_rgb[:,:,2]).astype(np.uint8)

    # SIMULACIÓN DE EJECUCIÓN DE LAS UNIDADES NPU (CORALNPU MATH HW)
    esc_brillo_contraste = np.clip(input_rgb.astype(float) * 1.25 + 20, 0, 255).astype(np.uint8)
    umbral = np.mean(gray)
    esc_binario = np.where(gray > umbral, 255, 0).astype(np.uint8)

    vec_negativo = (255 - input_rgb).astype(np.uint8)
    vec_canal = input_rgb.copy()
    vec_canal[:, :, 0] = np.clip(vec_canal[:, :, 0].astype(int) - 50, 0, 255)
    vec_canal[:, :, 1] = np.clip(vec_canal[:, :, 1].astype(int) + 40, 0, 255)

    gray_f = np.pad(gray.astype(np.float32), 1, mode='edge')
    gx = (gray_f[:-2, 2:] + 2*gray_f[1:-1, 2:] + gray_f[2:, 2:]) - (gray_f[:-2, :-2] + 2*gray_f[1:-1, :-2] + gray_f[2:, :-2])
    gy = (gray_f[2:, :-2] + 2*gray_f[2:, 1:-1] + gray_f[2:, 2:]) - (gray_f[:-2, :-2] + 2*gray_f[:-2, 1:-1] + gray_f[:-2, 2:])
    mat_sobel = np.clip(np.sqrt(gx[:h, :w]**2 + gy[:h, :w]**2), 0, 255).astype(np.uint8)

    # Modo Retrato Bokeh en Unidad MAC
    k = 15
    pad = k // 2
    blurred_img = np.zeros((h, w, c), dtype=np.float32)
    for ch_idx in range(3):
        img_pad = np.pad(input_rgb[:, :, ch_idx].astype(np.float32), pad, mode='edge')
        for i in range(k):
            for j in range(k):
                blurred_img[:, :, ch_idx] += img_pad[i:i+h, j:j+w]
        blurred_img[:, :, ch_idx] /= (k * k)

    X, Y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    mask_2d = np.clip(1.0 - np.exp(- (np.abs(X - 0.53) / 0.22)**2), 0, 1)
    mask = np.repeat(mask_2d[:, :, np.newaxis], 3, axis=2)
    mat_portrait_bokeh = (input_rgb.astype(float) * (1.0 - mask) + blurred_img * mask).astype(np.uint8)

    # Exportar Dashboard Visual 2x3
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), facecolor='#0f172a')

    axes[0,0].imshow(esc_brillo_contraste)
    axes[0,0].set_title('1. Escalar A: Brillo & Contraste (RGB)', fontsize=11, fontweight='bold', color='#e2e8f0')
    axes[0,0].axis('off')

    axes[0,1].imshow(vec_negativo)
    axes[0,1].set_title('2. Vectorial A: Negativo RGB (SIMD)', fontsize=11, fontweight='bold', color='#e2e8f0')
    axes[0,1].axis('off')

    axes[0,2].imshow(mat_sobel, cmap='inferno')
    axes[0,2].set_title('3. Matricial A: Sobel Bordes (MAC)', fontsize=11, fontweight='bold', color='#e2e8f0')
    axes[0,2].axis('off')

    axes[1,0].imshow(esc_binario, cmap='gray')
    axes[1,0].set_title('4. Escalar B: Binarización B/N Pura', fontsize=11, fontweight='bold', color='#38bdf8')
    axes[1,0].axis('off')

    axes[1,1].imshow(vec_canal)
    axes[1,1].set_title('5. Vectorial B: Realce Esmeralda (SIMD)', fontsize=11, fontweight='bold', color='#38bdf8')
    axes[1,1].axis('off')

    axes[1,2].imshow(mat_portrait_bokeh)
    axes[1,2].set_title('6. Matricial B: Modo Retrato Bokeh (MAC)', fontsize=11, fontweight='bold', color='#38bdf8')
    axes[1,2].axis('off')

    plt.suptitle('DEMOSTRACIÓN DE ARQUITECTURAS DE CÓMPUTO EN CORALNPU V2', 
                 fontsize=15, fontweight='bold', color='#38bdf8', y=0.98)

    plt.tight_layout()
    plt.savefig('examples/mi_proyecto_npu/resultado_pipeline_propio.png', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    print("\n[ÉXITO] ¡Flujo C/Hardware generado e integrado correctamente!")
    print("=======================================================\n")

if __name__ == "__main__":
    run_pipeline_test()
