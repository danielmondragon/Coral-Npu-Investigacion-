// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Cabecera oficial de intrinsecos vectoriales RISC-V (RVV).
#include <riscv_vector.h>
#include <string.h>

// Dos arreglos de 1024 enteros de 8 bits de entrada y una salida de 16 bits para no desbordar.
int8_t input_1[1024];
int8_t input_2[1024];
int16_t output[1024];

int main() {
  // Llenamos input_1 con 1s e input_2 con 6s para esperar 7 en la suma.
  memset(input_1, 1, 1024);
  memset(input_2, 6, 1024);
  const int8_t* input1_ptr = &input_1[0];
  const int8_t* input2_ptr = &input_2[0];
  int16_t* output_ptr = &output[0];

  // Recorremos los 1024 elementos avanzando de 32 en 32 con instrucciones vectoriales.
  for (int idx = 0; (idx + 31) < 1024; idx += 32) {
    // Cargamos 32 elementos int8 en registros vectoriales (LMUL=4).
    vint8m4_t input_v2 = __riscv_vle8_v_i8m4(input2_ptr + idx, 32);
    vint8m4_t input_v1 = __riscv_vle8_v_i8m4(input1_ptr + idx, 32);

    // Sumamos ambos vectores ampliando el resultado de 8 a 16 bits (LMUL=8).
    vint16m8_t temp_sum = __riscv_vwadd_vv_i16m8(input_v1, input_v2, 32);
    // Guardamos los 32 resultados de 16 bits en el arreglo de salida.
    __riscv_vse16_v_i16m8(output_ptr + idx, temp_sum, 32);
  }

  return 0;
}
