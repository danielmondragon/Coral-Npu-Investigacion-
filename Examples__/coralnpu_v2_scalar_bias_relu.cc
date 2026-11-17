#include "coralnpu_math.h"

float input_data[8] __attribute__((section(".data"))) = {-2.5f, 1.0f, -0.5f, 3.0f, -4.0f, 2.0f, 0.0f, -1.2f};
float bias_val      __attribute__((section(".data"))) = 0.5f;
float output_data[8] __attribute__((section(".data")));

int main() {
    for (int i = 0; i < 8; i++) {
        output_data[i] = scalar_bias_relu_op(input_data[i], bias_val);
    }
    return 0;
}
