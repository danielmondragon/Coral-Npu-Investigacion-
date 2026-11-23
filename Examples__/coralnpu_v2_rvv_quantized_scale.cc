#include "coralnpu_math.h"
#include <cstring>

// Arreglos de 128 elementos en DTCM (.data): pixeles y factores en int8, salida ampliada en int16.
int8_t  pixel_inputs[128]   __attribute__((section(".data")));
int8_t  scale_factors[128]  __attribute__((section(".data")));
int16_t scaled_outputs[128] __attribute__((section(".data")));

int main() {
    // Cargamos valores de prueba (4 y 3) para obtener 12 en cada posicion.
    memset(pixel_inputs, 4, 128);
    memset(scale_factors, 3, 128);

    // Subdivisión SIMD: avanza de 32 en 32 procesando bloques en paralelo
    for (int idx = 0; idx < 128; idx += 32) {
        coral_vreg_i8_t v_in = custom_vle8(pixel_inputs + idx);
        coral_vreg_i8_t v_scale = custom_vle8(scale_factors + idx);

        // Multiplica y amplía precisión de 8 a 16 bits (int8 * int8 -> int16)
        coral_vreg_i16_t v_prod = custom_vwmul(v_in, v_scale);

        // Copiamos el bloque multiplicado al arreglo de salida en DTCM.
        for (int k = 0; k < 32; k++) {
            scaled_outputs[idx + k] = v_prod.data[k];
        }
    }
    return 0;
}
