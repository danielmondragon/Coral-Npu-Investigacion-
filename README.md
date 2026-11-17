```markdown
# Investigación y Aplicación en Coral NPU V2 (RISC-V Vectorial)

En este repositorio guardamos toda la documentación, los códigos y los ejemplos que hicimos para probar y entender cómo funciona por dentro la arquitectura heterogénea de la **Coral NPU V2** (procesador RISC-V RV32 con unidad escalar/FPU, extensión vectorial RVV v1.0 y extensión matricial VME/Zvt).

## 👥 Integrantes
- Alejandra Calderón Zambrana
- Daniel Esteban Mondragón Juez  
*Departamento de Ingeniería Electrónica — Escuela Colombiana de Ingeniería Julio Garavito (Bogotá D.C., Colombia)*

---

## 📋 Cómo organizamos nuestros archivos

Fuimos ordenando todo el trabajo de la siguiente manera:

```text
Coral-Npu-Investigacion/
├── README.md                                  ← este archivo
├── docs/                                      ← 📄 LOS 3 PDF (recomendamos leerlos en orden 1 → 2 → 3)
│   ├── 01_Guia_Instalacion_CoralNPU.pdf
│   ├── 02_Informe_Ejemplos_CoralNPU.pdf
│   └── 03_Documentacion_Completa_CoralNPU.pdf
└── examples/                                  ← carpeta que llevamos a ~/coralnpu/examples
    ├── BUILD.bazel                            ← reglas Bazel completas: 2 oficiales + 3 propios
    ├── coralnpu_math.h                        ← nuestra librería base (tipos y operaciones MAC/Zvt)
    ├── hello_world_add_floats.cc              ← [oficial O1] suma escalar de flotantes FP32
    ├── rvv_add_intrinsic.cc                   ← [oficial O2] suma vectorial con intrínsecos RVV
    ├── o2_rvv_cocotb_test.py                  ← testbench Cocotb/Verilator para O2
    ├── coralnpu_v2_scalar_bias_relu.cc        ← [propio P1] prueba escalar: Bias + ReLU
    ├── p1_scalar_cocotb_test.py               ← testbench Cocotb/Verilator para P1
    ├── coralnpu_v2_rvv_quantized_scale.cc     ← [propio P2] prueba vectorial: escalado cuantizado
    ├── coralnpu_v2_matrix_mac_dot.cc          ← [propio P3] prueba matricial: producto punto MAC 4x4
    ├── zvt_matrix_mac_test.py                 ← testbench Cocotb/Verilator + auditoría ASM Zvt para P3
    └── mi_proyecto_npu/                       ← [propio P4] proyecto final con imágenes (128x128)
        ├── BUILD                              ← reglas Bazel del pipeline de visión
        ├── main.cc                            ← junta las 3 etapas (Escalar/FPU, RVV y VME/Zvt)
        ├── coralnpu_math.h                    ← funciones de nuestros filtros
        ├── input_128x128.npy                  ← imagen de prueba en escala de grises (16 KB para DTCM)
        ├── testbench_pipeline_img.py          ← testbench Cocotb: carga la foto en DTCM y arma el resultado
        ├── resultados_hw_zvt.npz              ← datos crudos extraídos de la memoria DTCM tras simular
        └── resultado_pipeline_propio.png      ← imagen final con las 6 etapas aplicadas

```

---

## 📚 En qué orden revisar nuestros PDF (`docs/`)

Para que se entienda bien todo lo que hicimos desde el primer día hasta lograr procesar imágenes completas en RTL, dejamos los tres informes en `docs/` pensados para leerse en orden:

1. **`01_Guia_Instalacion_CoralNPU.pdf`**
* **Lo que pusimos aquí:** Anotamos el paso a paso de cómo instalamos y dejamos listo el entorno en Linux / WSL2, el compilador LLVM de RISC-V, Bazel y el simulador Verilator / Cocotb.
* **Para tener en cuenta:** Si van a correr nuestros códigos, asegúrense de tener la carpeta del proyecto en `~/coralnpu` con las rutas que dejamos en esta guía.


2. **`02_Informe_Ejemplos_CoralNPU.pdf`**
* **Lo que pusimos aquí:** Explicamos con nuestras palabras cómo trabajan las tres partes principales de la NPU (Escalar, Vectorial y Matricial). Mostramos cómo guardamos los arreglos en la memoria DTCM (32 KB), comentamos cada código y pusimos las evidencias de las simulaciones pasando en Verilator.
* **Para tener en cuenta:** Fíjense en cómo mapeamos los datos en `.data`/`.bss` dentro de DTCM y en las cuentas manuales que hicimos para comprobar que cada ejercicio diera exactamente cero error frente a Python/NumPy.


3. **`03_Documentacion_Completa_CoralNPU.pdf`**
* **Lo que pusimos aquí:** Reunimos toda la investigación en un documento técnico explicando cómo es el hardware de la Coral NPU V2 por dentro (ITCM, DTCM, buses IBus/DBus/AXI4), las dos rutas de compilación (ELF bare-metal y MLIR/IREE) y nuestras conclusiones.
* **Para tener en cuenta:** Revisen las tablas comparativas y los comandos de auditoría binaria (`llvm-readelf`, `llvm-objdump`) al final.



---

## ⚙️ Qué hacer antes de correr los códigos

Para probar nuestros ejemplos en su computador, primero hay que copiar lo que dejamos en `examples/` dentro de la carpeta `examples/` del repositorio principal de Coral NPU:

```bash
cp -r examples/* ~/coralnpu/examples/
cd ~/coralnpu

```

---

## 🚀 Cómo probar cada ejercicio

### 1. Ejemplos Oficiales Base (`O1` y `O2`)

Empezamos compilando y simulando estos dos ejemplos oficiales del repositorio para comprobar que el compilador RISC-V, Bazel y Cocotb nos funcionaban bien:

```bash
bazel build //examples:hello_world_add_floats
bazel test --test_output=all //examples:o2_rvv_cocotb_test

```

* **¿Qué revisar?:** Que Bazel termine sin errores (`Build completed successfully`), genere los archivos (`.elf`, `.bin`, `.vmem`) en `bazel-bin/examples/` y en O2 la suma vectorial ampliada de 1024 elementos (`1 + 6 = 7`) pase en Verilator.

---

### 2. Nuestro Ejercicio 1 — Parte Escalar (`coralnpu_v2_scalar_bias_relu.cc`)

En esta primera prueba propia quisimos ver cómo trabaja la unidad escalar y la FPU con decimales (`FP32`): tomamos un arreglo de 8 flotantes, les sumamos un sesgo ($b = 0.5$) y aplicamos $\text{ReLU}(x) = \max(0, x + b)$ para dejar en cero los negativos.

* **Para compilar y correr la simulación RTL en Verilator + Cocotb:**

```bash
bazel test --test_output=all //examples:p1_scalar_cocotb_test

```

* **Lo que hay que mirar:**
* Guardamos los arreglos directamente en la memoria **DTCM** (`0x00010000`) para leerlos desde Python con `lookup_symbol` y `read_word`.
* En la consola de Verilator se ve cómo el vector `[-2.5, 1.0, -0.5, 3.0, -4.0, 2.0, 0.0, -1.2]` sale transformado exactamente en `[0.0, 1.5, 0.0, 3.5, 0.0, 2.5, 0.5, 0.0]` en **$525.00\text{ ns}$** de tiempo simulado con estado **`PASS`**.



---

### 3. Nuestro Ejercicio 2 — Parte Vectorial (`coralnpu_v2_rvv_quantized_scale.cc`)

Aquí probamos las instrucciones vectoriales de RISC-V (RVV) procesando bloques en paralelo. Trabajamos con 128 elementos de 8 bits (`int8_t`), ampliándolos temporalmente a 16 bits (`int16_t`) mediante `__riscv_vwmul_vv_i16m8` mientras se multiplican ($3 \times 4 = 12$) para evitar desbordamientos (*overflow*).

* **Para compilarlo:**

```bash
bazel build //examples:coralnpu_v2_rvv_quantized_scale

```

* **Lo que hay que mirar:**
* Cómo usamos los intrínsecos de `<riscv_vector.h>` ajustando dinámicamente el tamaño de bloque con `vsetvl` sin desbordar los registros de 128 bits (`VLEN = 128`).



---

### 4. Nuestro Ejercicio 3 — Parte Matricial (`coralnpu_v2_matrix_mac_dot.cc`)

En este tercer código probamos el núcleo matemático de la NPU: las operaciones de multiplicación y acumulación (**MAC**) usando nuestra librería `coralnpu_math.h` y la extensión matricial **VME/Zvt**. Con esto calculamos productos punto de matrices $4 \times 4$.

* **Para compilar, auditar el ensamblador Zvt y correr Verilator:**

```bash
bazel test --test_output=all //examples:zvt_matrix_mac_test

```

* **Lo que hay que mirar:**
* El testbench verifica tanto los resultados numéricos en la memoria **DTCM** como la presencia real de las instrucciones matriciales Zvt dentro del ejecutable `.elf` compilado.



---

### 5. Nuestro Proyecto Final — Pipeline de Visión de 6 Etapas (`examples/mi_proyecto_npu/`)

Para cerrar el trabajo armamos un sistema completo que junta las **3 unidades de la NPU** (Escalar/FPU, Vectorial RVV y Matricial VME/Zvt) sobre una imagen real de **$128 \times 128$ píxeles** (`input_128x128.npy`). Escogimos $128 \times 128$ porque ocupa $16\,384\text{ bytes}$ ($16\text{ KB}$), permitiéndonos alojar los búferes dentro de los **$32\text{ KB}$** físicos de la memoria **DTCM** sin desbordarla.

* **Para compilar y correr la simulación completa en Verilator + Cocotb:**

```bash
bazel test --test_output=all //examples/mi_proyecto_npu:testbench_pipeline_img

```

#### ¿Qué hicimos en las 6 etapas del pipeline?

1. **Carga AXI4 desde Python:** El testbench `testbench_pipeline_img.py` inyecta los $16\,384$ píxeles de `input_128x128.npy` directo en DTCM.
2. **Preprocesamiento Vectorial (RVV):** Ajuste de brillo/contraste y umbralización en bloques paralelos de 16 píxeles (`e8m1`).
3. **Convolución Espacial 2D (VME/Zvt):** Aplicación de filtros de detección de bordes mediante acumulación matricial MAC.
4. **Activación y Normalización Escalar (FPU):** Cálculo de estadísticas globales (media y varianza) y normalización en punto flotante `FP32`.
5. **Postprocesamiento Vectorial (RVV):** Saturación y empaquetado final a `uint8_t`.
6. **Extracción y Validación:** Python lee la DTCM palabra por palabra, guarda los datos en `resultados_hw_zvt.npz`, genera la figura `resultado_pipeline_propio.png` y compara contra NumPy obteniendo **exactitud bit a bit ($\text{MSE} = 0.0$)**.

---

## 🛠️ Problemas que nos salieron y cómo arreglarlos rápido

* **Si la memoria DTCM se llena al cambiar el tamaño de la imagen:** Recuerden que Coral NPU V2 trae **32 KB** de DTCM (`0x00010000`) y **8 KB** de ITCM (`0x00000000`). Si meten arreglos muy grandes en `.data` o `.bss`, el enlazador (`lld`) avisará que se pasó del límite; por eso trabajamos a $128 \times 128$ ($16\text{ KB}$).
* **Si Python dice `ModuleNotFoundError`:** Instalen las librerías necesarias en WSL2/Linux con:
```bash
pip install numpy pillow matplotlib cocotb

```


* **Para revisar si el compilador realmente usó instrucciones RVV o Zvt:** Pueden auditar cualquier `.elf` generado en `bazel-bin/` ejecutando:
```bash
llvm-objdump -d bazel-bin/examples/mi_proyecto_npu/pipeline_img_elf.elf | less

```



```

```
