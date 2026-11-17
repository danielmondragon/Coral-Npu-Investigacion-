#ifndef CORALNPU_MATH_H
#define CORALNPU_MATH_H

#include <cstdint>

typedef struct {
    int8_t data[32];
} coral_vreg_i8_t;

typedef struct {
    int16_t data[32];
} coral_vreg_i16_t;

typedef struct {
    int32_t data[4];
} coral_mreg_i32_t;

// Función escalar propia: Ajuste de Bias + Activación ReLU
inline float scalar_bias_relu_op(float val, float bias) {
    float res = val + bias;
    return (res < 0.0f) ? 0.0f : res;
}

// Funciones para los siguientes ejemplos
inline coral_vreg_i8_t custom_vle8(const int8_t* ptr) {
    coral_vreg_i8_t reg;
    for (int i = 0; i < 32; i++) {
        reg.data[i] = ptr[i];
    }
    return reg;
}

inline coral_vreg_i16_t custom_vwmul(coral_vreg_i8_t v1, coral_vreg_i8_t v2) {
    coral_vreg_i16_t res;
    for (int i = 0; i < 32; i++) {
        res.data[i] = ((int16_t)v1.data[i]) * ((int16_t)v2.data[i]);
    }
    return res;
}

inline coral_mreg_i32_t custom_vwmacc(coral_mreg_i32_t acc, coral_vreg_i8_t w, coral_vreg_i8_t f) {
    for (int row = 0; row < 4; row++) {
        int32_t dot_prod = 0;
        for (int col = 0; col < 4; col++) {
            dot_prod += ((int32_t)w.data[row * 4 + col]) * ((int32_t)f.data[row * 4 + col]);
        }
        acc.data[row] += dot_prod;
    }
    return acc;
}

#endif // CORALNPU_MATH_H

