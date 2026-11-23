#ifndef CORALNPU_MATH_H
#define CORALNPU_MATH_H

#include <cstdint>

// Registro vectorial simulado de 32 elementos de 8 bits (int8_t).
typedef struct {
    int8_t data[32];
} coral_vreg_i8_t;

// Registro vectorial simulado de 32 elementos de 16 bits (int16_t) para evitar desbordamientos.
typedef struct {
    int16_t data[32];
} coral_vreg_i16_t;

// Registro acumulador matricial de 4 elementos de 32 bits (int32_t).
typedef struct {
    int32_t data[4];
} coral_mreg_i32_t;

// Funcion escalar propia: suma un sesgo (bias) y aplica ReLU (recorta negativos a 0.0f).
inline float scalar_bias_relu_op(float val, float bias) {
    float res = val + bias;
    return (res < 0.0f) ? 0.0f : res;
}

// Carga un bloque de 32 datos int8 desde memoria hacia nuestra estructura vectorial.
inline coral_vreg_i8_t custom_vle8(const int8_t* ptr) {
    coral_vreg_i8_t reg;
    for (int i = 0; i < 32; i++) {
        reg.data[i] = ptr[i];
    }
    return reg;
}

// Multiplica elemento a elemento dos vectores int8 y amplia el resultado a int16.
inline coral_vreg_i16_t custom_vwmul(coral_vreg_i8_t v1, coral_vreg_i8_t v2) {
    coral_vreg_i16_t res;
    for (int i = 0; i < 32; i++) {
        res.data[i] = ((int16_t)v1.data[i]) * ((int16_t)v2.data[i]);
    }
    return res;
}

// Calcula productos punto de 4 elementos y los acumula en el registro de salida (operacion MAC).
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
