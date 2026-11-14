#include <stdio.h>
#include <stdint.h>
#include "coralnpu_math.h"

#define WIDTH 512
#define HEIGHT 512
#define TOTAL_PIXELS (WIDTH * HEIGHT)

// Buffers de entrada y salida ubicados en la memoria DTCM de la NPU
uint8_t input_rgb[TOTAL_PIXELS * 3];
uint8_t input_gray[TOTAL_PIXELS];

uint8_t out_esc_brillo[TOTAL_PIXELS * 3];
uint8_t out_esc_binario[TOTAL_PIXELS];
uint8_t out_vec_negativo[TOTAL_PIXELS * 3];
uint8_t out_vec_canal[TOTAL_PIXELS * 3];
uint8_t out_mat_sobel[TOTAL_PIXELS];

int main() {
    printf("[CoralNPU V2 Core] Iniciando procesamiento HW en memoria DTCM...\n");

    // 1. Etapa Escalar
    coralnpu_scalar_brightness_contrast(input_rgb, out_esc_brillo, TOTAL_PIXELS * 3, 1.25f, 20);
    coralnpu_scalar_binarize(input_gray, out_esc_binario, WIDTH, HEIGHT, 128);

    // 2. Etapa Vectorial SIMD
    coralnpu_simd_invert(input_rgb, out_vec_negativo, TOTAL_PIXELS * 3);
    coralnpu_simd_emerald_enhance(input_rgb, out_vec_canal, TOTAL_PIXELS);

    // 3. Etapa Matricial MAC
    coralnpu_mac_sobel_2d(input_gray, out_mat_sobel, WIDTH, HEIGHT);

    printf("[CoralNPU V2 Core] Procesamiento finalizado con éxito.\n");
    return 0;
}
