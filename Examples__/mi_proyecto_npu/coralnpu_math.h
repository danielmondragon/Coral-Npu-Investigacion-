#ifndef CORALNPU_MATH_H
#define CORALNPU_MATH_H

#include <stdint.h>
#include <math.h>

// 1. UNIDAD ESCALAR: Brillo, Contraste y Binarización elemento a elemento
// Ajusta brillo y contraste (alpha * pixel + beta) limitando el resultado al rango [0, 255].
static inline void coralnpu_scalar_brightness_contrast(const uint8_t* in, uint8_t* out, int size, float alpha, int beta) {
    for (int i = 0; i < size; i++) {
        int val = (int)(in[i] * alpha) + beta;
        out[i] = (uint8_t)(val > 255 ? 255 : (val < 0 ? 0 : val));
    }
}

// Convierte una imagen en escala de grises a blanco y negro (0 o 255) segun un umbral.
static inline void coralnpu_scalar_binarize(const uint8_t* in_gray, uint8_t* out, int width, int height, uint8_t threshold) {
    int total = width * height;
    for (int i = 0; i < total; i++) {
        out[i] = (in_gray[i] > threshold) ? 255 : 0;
    }
}

// 2. UNIDAD VECTORIAL (SIMD): Operaciones concurrentes multi-canal RGB
// Invierte los colores restando cada valor de 255.
static inline void coralnpu_simd_invert(const uint8_t* in, uint8_t* out, int size) {
    for (int i = 0; i < size; i++) {
        out[i] = 255 - in[i];
    }
}

// Realza el tono esmeralda reduciendo el canal rojo (R - 50) y aumentando el verde (G + 40).
static inline void coralnpu_simd_emerald_enhance(const uint8_t* in_rgb, uint8_t* out_rgb, int total_pixels) {
    for (int i = 0; i < total_pixels; i++) {
        int idx = i * 3;
        int r = (int)in_rgb[idx] - 50;
        int g = (int)in_rgb[idx + 1] + 40;
        int b = (int)in_rgb[idx + 2];
        out_rgb[idx]     = (uint8_t)(r < 0 ? 0 : r);
        out_rgb[idx + 1] = (uint8_t)(g > 255 ? 255 : g);
        out_rgb[idx + 2] = (uint8_t)(b > 255 ? 255 : b);
    }
}

// 3. UNIDAD MATRICIAL (MAC): Convoluciones 2D y Máscara de Profundidad Bokeh
// Calcula la magnitud de bordes Sobel en 2D (gradientes gx y gy) sobre una imagen en grises.
static inline void coralnpu_mac_sobel_2d(const uint8_t* in_gray, uint8_t* out_sobel, int w, int h) {
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int gx = (-1 * in_gray[(y-1)*w + (x-1)]) + (1 * in_gray[(y-1)*w + (x+1)]) +
                     (-2 * in_gray[y*w + (x-1)])     + (2 * in_gray[y*w + (x+1)]) +
                     (-1 * in_gray[(y+1)*w + (x-1)]) + (1 * in_gray[(y+1)*w + (x+1)]);

            int gy = (-1 * in_gray[(y-1)*w + (x-1)]) + (-2 * in_gray[(y-1)*w + x]) + (-1 * in_gray[(y-1)*w + (x+1)]) +
                     ( 1 * in_gray[(y+1)*w + (x-1)]) + ( 2 * in_gray[(y+1)*w + x]) + ( 1 * in_gray[(y+1)*w + (x+1)]);

            int mag = (int)sqrtf((float)(gx * gx + gy * gy));
            out_sobel[y*w + x] = (uint8_t)(mag > 255 ? 255 : mag);
        }
    }
}

#endif // CORALNPU_MATH_H
