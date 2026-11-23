#include <riscv_vector.h>
#include <cstdint>

#define BATCH_PIXELS 64  // Procesamos la imagen en lotes de 64 pixeles para cuidar la memoria DTCM.
#define TE 16            // Tamano base del bloque (tile) en la unidad matricial Zvt.

// Constantes de configuracion y codigos de instruccion para la extension matricial Zvt.
static constexpr uint32_t kVtypeSew8Lmul1  = 0xC0;
static constexpr uint32_t kVtypeSew32Lmul4 = 0xD2;
static constexpr uint32_t kZvtOpcodeOpV    = 0x57u;
static constexpr uint32_t kZvtOpcodeOpVe   = 0x77u;

static constexpr uint32_t ZvtWord(uint32_t funct6, uint32_t vs2, uint32_t rs1,
                                  uint32_t funct3, uint32_t rd, uint32_t opcode) {
  return (funct6 << 26) | (1u << 25) | (vs2 << 20) | (rs1 << 15) |
         (funct3 << 12) | (rd << 7) | (opcode & 0x7Fu);
}

static constexpr uint32_t kZvtVtmmsTile0  = ZvtWord(0x3C, 8, 16, 0, 1, kZvtOpcodeOpVe);
static constexpr uint32_t kZvtVtzeroTile0 = ZvtWord(0x10, 30, 0, 6, 0, kZvtOpcodeOpV);
static constexpr uint32_t kZvtVtmvVT      = ZvtWord(0x10, 31, 10, 6, 4, kZvtOpcodeOpV);

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

// Raiz cuadrada entera rapida para calcular la magnitud del gradiente sin depender de librerias pesadas.
static inline uint32_t fast_isqrt(uint32_t x) {
  uint32_t res = 0;
  uint32_t bit = 1u << 30;
  while (bit > x) bit >>= 2;
  while (bit != 0) {
    if (x >= res + bit) {
      x -= res + bit;
      res = (res >> 1) + bit;
    } else {
      res >>= 1;
    }
    bit >>= 2;
  }
  return res;
}

// Buffers compartidos en memoria DTCM (accesibles por Cocotb vía AXI)
alignas(16) uint8_t dtcm_in_rgb[BATCH_PIXELS * 3] __attribute__((section(".data")));
alignas(16) uint8_t dtcm_in_gray[BATCH_PIXELS] __attribute__((section(".data")));
volatile uint32_t dtcm_gray_thresh __attribute__((section(".data"))) = 128;

// Salidas Etapa 1: FPU Escalar
alignas(16) uint8_t dtcm_out_esc_brillo[BATCH_PIXELS * 3] __attribute__((section(".data")));
alignas(16) uint8_t dtcm_out_esc_bin[BATCH_PIXELS] __attribute__((section(".data")));

// Salidas Etapa 2: Vectorial RVV (<riscv_vector.h>)
alignas(16) uint8_t dtcm_out_rvv_neg[BATCH_PIXELS * 3] __attribute__((section(".data")));
alignas(16) uint8_t dtcm_out_rvv_esmeralda[BATCH_PIXELS * 3] __attribute__((section(".data")));

// Entradas y salidas Etapa 3: Matricial Sistólica Zvt (64 parches 3x3 por lote)
alignas(16) uint8_t dtcm_zvt_patches[BATCH_PIXELS * 9] __attribute__((section(".data")));
alignas(16) int8_t dtcm_zvt_kernels[9 * 3] __attribute__((section(".data")));
alignas(16) uint8_t dtcm_out_zvt_sobel[BATCH_PIXELS] __attribute__((section(".data")));
alignas(16) uint8_t dtcm_out_zvt_blur[BATCH_PIXELS] __attribute__((section(".data")));

// Buffers internos alineados para carga Zvt
alignas(16) int8_t zvt_a_slot[2 * TE] __attribute__((section(".data")));
alignas(16) uint8_t zvt_b_slot[2 * TE] __attribute__((section(".data")));
alignas(16) int32_t zvt_tile_tmp[TE * TE] __attribute__((section(".data")));

// =============================================================================
// ETAPA 1: PROCESAMIENTO ESCALAR EN FPU (flw, fmul.s, fadd.s, fsw)
// =============================================================================
__attribute__((noinline)) void run_stage1_scalar_fpu() {
  // Ajustamos brillo y contraste (1.25 * x + 20) saturando entre 0 y 255.
  const float alpha = 1.25f;
  const float beta = 20.0f;
  for (int i = 0; i < BATCH_PIXELS * 3; ++i) {
    float val = static_cast<float>(dtcm_in_rgb[i]) * alpha + beta;
    if (val > 255.0f) val = 255.0f;
    if (val < 0.0f) val = 0.0f;
    dtcm_out_esc_brillo[i] = static_cast<uint8_t>(val);
  }

  // Binarizamos la escala de grises comparando cada pixel contra el umbral recibido.
  uint8_t th = static_cast<uint8_t>(dtcm_gray_thresh);
  for (int i = 0; i < BATCH_PIXELS; ++i) {
    dtcm_out_esc_bin[i] = (dtcm_in_gray[i] > th) ? 255 : 0;
  }
}

// =============================================================================
// ETAPA 2: PROCESAMIENTO VECTORIAL SIMD REAL (<riscv_vector.h>)
// =============================================================================
__attribute__((noinline)) void run_stage2_rvv_simd() {
  // Negativo RGB en paralelo usando resta vectorial (255 - pixel).
  size_t n_rgb = BATCH_PIXELS * 3;
  size_t idx = 0;
  while (idx < n_rgb) {
    size_t vl = __riscv_vsetvl_e8m4(n_rgb - idx);
    vuint8m4_t v_in = __riscv_vle8_v_u8m4(&dtcm_in_rgb[idx], vl);
    vuint8m4_t v_neg = __riscv_vrsub_vx_u8m4(v_in, 255, vl);
    __riscv_vse8_v_u8m4(&dtcm_out_rvv_neg[idx], v_neg, vl);
    idx += vl;
  }

  // Filtro esmeralda segmentado por canales R, G y B con saturacion vectorial.
  size_t p = 0;
  while (p < BATCH_PIXELS) {
    size_t vl = __riscv_vsetvl_e8m1(BATCH_PIXELS - p);
    vuint8m1x3_t rgb = __riscv_vlseg3e8_v_u8m1x3(&dtcm_in_rgb[p * 3], vl);
    vuint8m1_t r = __riscv_vget_v_u8m1x3_u8m1(rgb, 0);
    vuint8m1_t g = __riscv_vget_v_u8m1x3_u8m1(rgb, 1);
    vuint8m1_t b = __riscv_vget_v_u8m1x3_u8m1(rgb, 2);

    r = __riscv_vssubu_vx_u8m1(r, 50, vl);
    g = __riscv_vsaddu_vx_u8m1(g, 40, vl);

    vuint8m1x3_t out_rgb = __riscv_vcreate_v_u8m1x3(r, g, b);
    __riscv_vsseg3e8_v_u8m1x3(&dtcm_out_rvv_esmeralda[p * 3], out_rgb, vl);
    p += vl;
  }
}

// =============================================================================
// ETAPA 3: PROCESAMIENTO MATRICIAL SISTÓLICO ZVT (vtmms.tvv / VmeCoreMiniAxi)
// =============================================================================
__attribute__((noinline)) void run_stage3_zvt_matrix() {
  // Dividimos los 64 parches 3x3 del lote en 4 sub-bloques de 16 parches para el Tile Zvt.
  for (int sub = 0; sub < 4; ++sub) {
    const uint8_t *patches = &dtcm_zvt_patches[sub * 16 * 9];

    // Limpiamos el acumulador matricial antes de iniciar el sub-bloque.
    vme_msetmtype(MtypeValue(TE, 1, 1), kVtypeSew32Lmul4);
    (void)vme_msettn(TE);
    asm volatile(".word %0" :: "i"(kZvtVtzeroTile0) : "memory");

    // Acumulamos los 9 pesos del kernel 3x3 en pasos de a 2 (tk=2) sobre Tile 0.
    for (int k = 0; k < 9; k += 2) {
      int cur_tk = (k + 1 < 9) ? 2 : 1;
      for (int row = 0; row < 3; ++row) {
        zvt_a_slot[0 * TE + row] = dtcm_zvt_kernels[k * 3 + row];
        zvt_a_slot[1 * TE + row] = (cur_tk == 2) ? dtcm_zvt_kernels[(k + 1) * 3 + row] : 0;
      }
      for (int col = 0; col < 16; ++col) {
        zvt_b_slot[0 * TE + col] = patches[col * 9 + k];
        zvt_b_slot[1 * TE + col] = (cur_tk == 2) ? patches[col * 9 + (k + 1)] : 0;
      }

      vme_msetmtype(MtypeValue(3, cur_tk, 3), kVtypeSew8Lmul1);
      (void)vme_msettm(3);
      (void)vme_msettn(TE);
      asm volatile(
          "vle8.v v8,  (%0)
"
          "vle8.v v10, (%1)
"
          "vle8.v v16, (%2)
"
          "vle8.v v18, (%3)
"
          :: "r"(&zvt_a_slot[0]), "r"(&zvt_a_slot[TE]),
             "r"(&zvt_b_slot[0]), "r"(&zvt_b_slot[TE])
          : "v8", "v10", "v16", "v18", "memory");
      (void)vme_msettn(16);
      asm volatile(".word %0" :: "i"(kZvtVtmmsTile0) : "memory");
    }

    // Descargamos las 3 filas calculadas (Gx, Gy y desenfoque) desde Tile 0 hacia DTCM.
    vme_msetmtype(MtypeValue(TE, 1, 1), kVtypeSew32Lmul4);
    (void)vme_msettn(TE);
    for (uint32_t r = 0; r < 3; ++r) {
      int32_t *row_ptr = &zvt_tile_tmp[r * TE];
      register uint32_t a0_arg asm("a0") = r;
      asm volatile(".word %0" :: "i"(kZvtVtmvVT), "r"(a0_arg) : "v4", "v5", "v6", "v7", "memory");
      asm volatile("vse32.v v4, (%0)" :: "r"(row_ptr) : "memory");
    }

    // Calculamos la magnitud del borde Sobel y el promedio del desenfoque para cada pixel.
    for (int p = 0; p < 16; ++p) {
      int out_idx = sub * 16 + p;
      int32_t gx = zvt_tile_tmp[0 * TE + p];
      int32_t gy = zvt_tile_tmp[1 * TE + p];
      uint32_t mag = fast_isqrt(static_cast<uint32_t>(gx * gx + gy * gy));
      if (mag > 255u) mag = 255u;
      dtcm_out_zvt_sobel[out_idx] = static_cast<uint8_t>(mag);

      int32_t blur_sum = zvt_tile_tmp[2 * TE + p] / 9;
      if (blur_sum > 255) blur_sum = 255;
      if (blur_sum < 0) blur_sum = 0;
      dtcm_out_zvt_blur[out_idx] = static_cast<uint8_t>(blur_sum);
    }
  }
}

// Ejecuta en orden las tres etapas sobre el lote actual cargado en DTCM.
int main() {
  run_stage1_scalar_fpu();
  run_stage2_rvv_simd();
  run_stage3_zvt_matrix();
  return 0;
}
