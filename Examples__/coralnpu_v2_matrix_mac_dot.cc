#include <cstdint>

// -----------------------------------------------------------------------------
// Definiciones nativas de hardware VME / Zvt (CoralNPU V2 - VmeCoreMiniAxi)
// -----------------------------------------------------------------------------
#define TE 16
static constexpr uint32_t kVtypeSew8Lmul1  = 0xC0;  // ta/ma, SEW8, LMUL1
static constexpr uint32_t kVtypeSew32Lmul4 = 0xD2;  // ta/ma, SEW32, LMUL4

static constexpr uint32_t kZvtOpcodeOpV  = 0x57u;
static constexpr uint32_t kZvtOpcodeOpVe = 0x77u;

static constexpr uint32_t ZvtWord(uint32_t funct6, uint32_t vs2, uint32_t rs1,
                                  uint32_t funct3, uint32_t rd, uint32_t opcode) {
  return (funct6 << 26) | (1u << 25) | (vs2 << 20) | (rs1 << 15) |
         (funct3 << 12) | (rd << 7) | (opcode & 0x7Fu);
}

// vtmms.tvv: int8 (A) x uint8 (B) -> int32 (Tile 0)
static constexpr uint32_t kZvtVtmmsTile0 = ZvtWord(0x3C, 8, 16, 0, 1, kZvtOpcodeOpVe);
// vtzero Tile 0
static constexpr uint32_t kZvtVtzeroTile0 = ZvtWord(0x10, 30, 0, 6, 0, kZvtOpcodeOpV);
// vtmv.v.t v4, a0
static constexpr uint32_t kZvtVtmvVT = ZvtWord(0x10, 31, 10, 6, 4, kZvtOpcodeOpV);

static constexpr uint32_t MtypeValue(uint32_t tm, uint32_t tk, uint32_t mtwiden) {
  return ((tm & 0x3FFF) << 10) | ((tk & 0x7) << 5) | (mtwiden & 0x3);
}

static inline __attribute__((always_inline)) void vme_msetmtype(uint32_t mtype, uint32_t vtype) {
  register uint32_t a0_arg asm("a0") = mtype;
  register uint32_t a1_arg asm("a1") = vtype;
  asm volatile(".insn r 0b1010111, 0b111, 0b1000001, x0, a0, a1" :: "r"(a0_arg), "r"(a1_arg));
}

static inline __attribute__((always_inline)) uint32_t vme_msettm(uint32_t tm) {
  register uint32_t a0_arg asm("a0") = tm;
  register uint32_t t0_out asm("t0");
  asm volatile(".insn r 0b1010111, 0b111, 0b1000010, t0, a0, x1" : "=r"(t0_out) : "r"(a0_arg));
  return t0_out;
}

static inline __attribute__((always_inline)) uint32_t vme_msettn(uint32_t tn) {
  register uint32_t a0_arg asm("a0") = tn;
  register uint32_t t0_out asm("t0");
  asm volatile(".insn r 0b1010111, 0b111, 0b1000010, t0, a0, x0" : "=r"(t0_out) : "r"(a0_arg));
  return t0_out;
}

// -----------------------------------------------------------------------------
// Matrices 4x4 originales (empaquetadas en slots TE=16 en memoria DTCM)
// -----------------------------------------------------------------------------
// weight_matrix 4x4:
//   [1, 2, 1, 0]
//   [0, 1, 3, 1]
//   [2, 0, 1, 2]
//   [1, 1, 0, 1]
alignas(16) int8_t weight_matrix[16] __attribute__((section(".data"))) = {
    1, 2, 1, 0,
    0, 1, 3, 1,
    2, 0, 1, 2,
    1, 1, 0, 1
};

// feature_vector 4x4:
//   [2, 1, 3, 1]
//   [1, 4, 1, 2]
//   [0, 2, 2, 1]
//   [3, 1, 0, 4]
alignas(16) uint8_t feature_vector[16] __attribute__((section(".data"))) = {
    2, 1, 3, 1,
    1, 4, 1, 2,
    0, 2, 2, 1,
    3, 1, 0, 4
};

// Buffers de paso empaquetados por columna (A) y fila (B) para Zvt (tk=2)
alignas(16) int8_t zvt_a_cols[4 * TE] __attribute__((section(".data"))) = {};
alignas(16) uint8_t zvt_b_rows[4 * TE] __attribute__((section(".data"))) = {};

// Tile acumulador completo (16x16 int32) y salida final 4x4 / diagonal MAC
alignas(16) int32_t zvt_tile_out[TE * TE] __attribute__((section(".data"))) = {};
alignas(16) int32_t matmul_4x4_out[16] __attribute__((section(".data"))) = {};
alignas(16) int32_t acc_output[4] __attribute__((section(".data"))) = {};

int main() {
  // 1. Limpiar registro Tile 0 del arreglo sistólico Zvt (vtzero)
  vme_msetmtype(MtypeValue(TE, 1, 1), kVtypeSew32Lmul4);
  (void)vme_msettn(TE);
  asm volatile(".word %0" :: "i"(kZvtVtzeroTile0) : "memory");

  // 2. Ejecutar multiplicación matricial C = A(4x4) * B(4x4) en 2 pasadas de tk=2
  // Pasada k = 0, 1
  for (int m = 0; m < 4; ++m) {
    zvt_a_cols[0 * TE + m] = weight_matrix[m * 4 + 0];
    zvt_a_cols[1 * TE + m] = weight_matrix[m * 4 + 1];
  }
  for (int n = 0; n < 4; ++n) {
    zvt_b_rows[0 * TE + n] = feature_vector[0 * 4 + n];
    zvt_b_rows[1 * TE + n] = feature_vector[1 * 4 + n];
  }

  vme_msetmtype(MtypeValue(4, 2, 3), kVtypeSew8Lmul1);
  (void)vme_msettm(4);
  (void)vme_msettn(TE);
  asm volatile(
      "vle8.v v8,  (%0)\n"
      "vle8.v v10, (%1)\n"
      "vle8.v v16, (%2)\n"
      "vle8.v v18, (%3)\n"
      :: "r"(&zvt_a_cols[0]), "r"(&zvt_a_cols[TE]),
         "r"(&zvt_b_rows[0]), "r"(&zvt_b_rows[TE])
      : "v8", "v10", "v16", "v18", "memory");
  (void)vme_msettn(4);
  asm volatile(".word %0" :: "i"(kZvtVtmmsTile0) : "memory");

  // Pasada k = 2, 3 (acumula directamente sobre el mismo Tile 0 de Zvt)
  for (int m = 0; m < 4; ++m) {
    zvt_a_cols[0 * TE + m] = weight_matrix[m * 4 + 2];
    zvt_a_cols[1 * TE + m] = weight_matrix[m * 4 + 3];
  }
  for (int n = 0; n < 4; ++n) {
    zvt_b_rows[0 * TE + n] = feature_vector[2 * 4 + n];
    zvt_b_rows[1 * TE + n] = feature_vector[3 * 4 + n];
  }

  vme_msetmtype(MtypeValue(4, 2, 3), kVtypeSew8Lmul1);
  (void)vme_msettm(4);
  (void)vme_msettn(TE);
  asm volatile(
      "vle8.v v8,  (%0)\n"
      "vle8.v v10, (%1)\n"
      "vle8.v v16, (%2)\n"
      "vle8.v v18, (%3)\n"
      :: "r"(&zvt_a_cols[0]), "r"(&zvt_a_cols[TE]),
         "r"(&zvt_b_rows[0]), "r"(&zvt_b_rows[TE])
      : "v8", "v10", "v16", "v18", "memory");
  (void)vme_msettn(4);
  asm volatile(".word %0" :: "i"(kZvtVtmmsTile0) : "memory");

  // 3. Descargar el registro Tile 0 del hardware Zvt hacia DTCM (vtmv.v.t + vse32.v)
  vme_msetmtype(MtypeValue(TE, 1, 1), kVtypeSew32Lmul4);
  (void)vme_msettn(TE);
  for (uint32_t r = 0; r < 4; ++r) {
    int32_t *row_ptr = &zvt_tile_out[r * TE];
    register uint32_t a0_arg asm("a0") = r;
    asm volatile(".word %0" :: "i"(kZvtVtmvVT), "r"(a0_arg) : "v4", "v5", "v6", "v7", "memory");
    asm volatile("vse32.v v4, (%0)" :: "r"(row_ptr) : "memory");
  }

  // Guardar la matriz producto 4x4 completa y la diagonal de productos punto
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      matmul_4x4_out[r * 4 + c] = zvt_tile_out[r * TE + c];
    }
    acc_output[r] = zvt_tile_out[r * TE + r];
  }

  return 0;
}
