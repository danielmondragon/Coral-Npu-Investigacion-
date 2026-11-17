# Investigación y Aplicación en Coral NPU V2 (RISC-V Vectorial)

En este repositorio almacenamos toda la documentación, los códigos y los ejemplos desarrollados para investigar, probar y comprender el funcionamiento interno de la arquitectura heterogénea de la **Coral NPU V2**.

La plataforma integra un procesador RISC-V RV32 con unidad escalar/FPU, extensión vectorial RVV v1.0 y extensión matricial VME/Zvt.

## 👥 Integrantes

- **Alejandra Calderón Zambrana**
- **Daniel Esteban Mondragón Juez**

**Departamento de Ingeniería Electrónica**  
Escuela Colombiana de Ingeniería Julio Garavito  
Bogotá D.C., Colombia.

---

## 📋 Organización del repositorio

Organizamos los archivos del proyecto de la siguiente manera:

```text
Coral-Npu-Investigacion/
│
├── README.md
│
├── docs/
│   ├── 01_Guia_Instalacion_CoralNPU.pdf
│   ├── 02_Informe_Ejemplos_CoralNPU.pdf
│   └── 03_Documentacion_Completa_CoralNPU.pdf
│
└── examples/
    ├── BUILD.bazel
    ├── coralnpu_math.h
    │
    ├── hello_world_add_floats.cc
    ├── rvv_add_intrinsic.cc
    ├── o2_rvv_cocotb_test.py
    │
    ├── coralnpu_v2_scalar_bias_relu.cc
    ├── p1_scalar_cocotb_test.py
    │
    ├── coralnpu_v2_rvv_quantized_scale.cc
    │
    ├── coralnpu_v2_matrix_mac_dot.cc
    ├── zvt_matrix_mac_test.py
    │
    └── mi_proyecto_npu/
        ├── BUILD
        ├── main.cc
        ├── coralnpu_math.h
        ├── input_128x128.npy
        ├── testbench_pipeline_img.py
        ├── resultados_hw_zvt.npz
        └── resultado_pipeline_propio.png
```

### Descripción de las carpetas

- **docs/**: contiene los tres documentos PDF que reúnen la investigación, instalación y resultados.
- **examples/**: contiene los ejemplos oficiales y los ejercicios desarrollados durante la investigación.
- **mi_proyecto_npu/**: contiene el proyecto final, en el que integramos las tres unidades de procesamiento para ejecutar un pipeline de visión sobre una imagen de 128 × 128 píxeles.

---

## 📚 Documentación del proyecto

Los tres documentos PDF están organizados para leerse en orden, desde la instalación del entorno hasta el análisis completo de la arquitectura.

### 01. Guía de instalación — `01_Guia_Instalacion_CoralNPU.pdf`

En este documento registramos el procedimiento que seguimos para instalar y configurar el entorno de desarrollo en Linux mediante WSL2.

Incluye:

- Configuración de Linux / WSL2.
- Instalación del compilador LLVM para RISC-V.
- Instalación y configuración de Bazel.
- Configuración de Verilator.
- Instalación y configuración de Cocotb.
- Preparación del repositorio principal de Coral NPU.

**Importante:** Para ejecutar nuestros códigos, es necesario tener el repositorio principal de Coral NPU ubicado en `~/coralnpu`, respetando las rutas y configuraciones descritas en la guía.

### 02. Informe de ejemplos — `02_Informe_Ejemplos_CoralNPU.pdf`

En este documento explicamos los ejercicios que desarrollamos para comprender las tres unidades principales de procesamiento de la Coral NPU V2:

1. Procesamiento escalar mediante la FPU.
2. Procesamiento vectorial mediante RVV.
3. Procesamiento matricial mediante VME/Zvt.

También incluimos:

- Explicación del funcionamiento de cada código.
- Organización y almacenamiento de datos en la memoria DTCM (32 KB).
- Ubicación de variables en las secciones `.data` y `.bss`.
- Resultados de las simulaciones RTL con Verilator y Cocotb.
- Comprobaciones matemáticas realizadas manualmente.
- Comparaciones de los resultados obtenidos por hardware frente a Python/NumPy.

### 03. Documentación completa — `03_Documentacion_Completa_CoralNPU.pdf`

Este documento reúne la investigación técnica sobre la arquitectura interna de la Coral NPU V2.

Incluye:

- Arquitectura general del procesador.
- Organización de las memorias ITCM y DTCM.
- Funcionamiento de los buses IBus, DBus y AXI4.
- Unidad escalar y FPU.
- Extensión vectorial RISC-V Vector (RVV).
- Extensión matricial VME/Zvt.
- Rutas de compilación ELF bare-metal y MLIR/IREE.
- Análisis de los ejecutables generados.
- Auditoría de instrucciones mediante `llvm-readelf` y `llvm-objdump`.
- Comparación de las diferentes unidades de procesamiento y conclusiones.

---

## ⚙️ Preparación del entorno

Para ejecutar nuestros ejemplos, primero debemos copiar los archivos de la carpeta `examples/` de este repositorio al directorio `examples/` del repositorio principal de Coral NPU.

Desde la terminal de WSL2/Linux:

```bash
cp -r examples/* ~/coralnpu/examples/
cd ~/coralnpu
```

Es importante realizar este procedimiento después de haber configurado correctamente el entorno siguiendo la guía de instalación.

---

## 🚀 Ejecución de los ejercicios

### 1. Ejemplos oficiales base (O1 y O2)

Comenzamos utilizando dos ejemplos oficiales del repositorio para verificar el funcionamiento del compilador RISC-V, Bazel y el entorno de simulación.

**O1 — Suma escalar de flotantes FP32**

Archivo: `hello_world_add_floats.cc`

Permite comprobar el funcionamiento del procesamiento escalar mediante operaciones con números de punto flotante de precisión simple (FP32).

Compilación:

```bash
bazel build //examples:hello_world_add_floats
```

**O2 — Suma vectorial mediante RVV**

Archivo: `rvv_add_intrinsic.cc`

Utiliza intrínsecos de la extensión vectorial RISC-V para realizar operaciones sobre múltiples elementos en paralelo.

Ejecución del testbench:

```bash
bazel test --test_output=all //examples:o2_rvv_cocotb_test
```

**¿Qué debemos verificar?**

- Que Bazel finalice correctamente la compilación.
- Que se generen los archivos ejecutables `.elf`, `.bin` y `.vmem`.
- Que la simulación en Verilator termine con estado `PASS`.
- Que la suma vectorial de los 1024 elementos produzca correctamente el resultado esperado: `1 + 6 = 7`.

---

### 2. Ejercicio propio P1 — Procesamiento escalar (Bias + ReLU)

Archivo: `coralnpu_v2_scalar_bias_relu.cc`

En este primer ejercicio propio buscamos comprobar el funcionamiento de la unidad escalar y la FPU mediante operaciones con números de punto flotante FP32.

Utilizamos un arreglo de ocho elementos al que le aplicamos dos operaciones:

1. Suma de un sesgo o *Bias* de valor \(b = 0.5\).
2. Función de activación ReLU:

\[
ReLU(x)=\max(0,x+b)
\]

El objetivo es sumar el sesgo a cada elemento y convertir en cero los resultados negativos.

**Ejecución de la simulación RTL:**

```bash
bazel test --test_output=all //examples:p1_scalar_cocotb_test
```

**¿Qué debemos verificar?**

Los arreglos se almacenan directamente en la memoria DTCM, cuya dirección base es `0x00010000`.

Desde Python utilizamos `lookup_symbol` y `read_word` para localizar y leer los datos almacenados en memoria.

Vector de entrada:

```text
[-2.5, 1.0, -0.5, 3.0, -4.0, 2.0, 0.0, -1.2]
```

Vector de salida esperado:

```text
[0.0, 1.5, 0.0, 3.5, 0.0, 2.5, 0.5, 0.0]
```

La simulación obtiene el resultado esperado en aproximadamente `525.00 ns` de tiempo simulado y finaliza con estado `PASS`.

---

### 3. Ejercicio propio P2 — Procesamiento vectorial (Escalado cuantizado)

Archivo: `coralnpu_v2_rvv_quantized_scale.cc`

En este ejercicio buscamos aprovechar las instrucciones de la extensión vectorial RVV para procesar varios datos simultáneamente.

Trabajamos con 128 elementos de tipo `int8_t` (enteros con signo de 8 bits).

Durante el procesamiento ampliamos los datos a 16 bits mediante el intrínseco:

```cpp
__riscv_vwmul_vv_i16m8
```

Esta operación realiza una multiplicación vectorial con ampliación de los resultados, evitando que las operaciones queden limitadas al rango de representación de 8 bits.

Por ejemplo:

\[
3 \times 4 = 12
\]

**Compilación:**

```bash
bazel build //examples:coralnpu_v2_rvv_quantized_scale
```

**¿Qué debemos verificar?**

- El uso de intrínsecos de la biblioteca `<riscv_vector.h>`.
- La configuración dinámica de la longitud vectorial mediante `vsetvl`.
- El procesamiento de los datos en bloques vectoriales.
- La ampliación de los resultados de 8 a 16 bits.
- La correcta utilización de los registros vectoriales con `VLEN = 128`.

---

### 4. Ejercicio propio P3 — Procesamiento matricial (MAC 4×4)

Archivo: `coralnpu_v2_matrix_mac_dot.cc`

En este tercer ejercicio buscamos probar las operaciones de multiplicación y acumulación (MAC) mediante la extensión matricial VME/Zvt.

Para ello desarrollamos una operación de producto punto matricial de dimensiones 4 × 4, utilizando nuestra biblioteca:

```cpp
coralnpu_math.h
```

El objetivo es comprobar el funcionamiento del procesamiento matricial y verificar que el compilador genere instrucciones correspondientes a la extensión Zvt.

**Compilación, simulación y auditoría:**

```bash
bazel test --test_output=all //examples:zvt_matrix_mac_test
```

**¿Qué debemos verificar?**

El testbench realiza dos comprobaciones principales:

1. **Validación numérica:** comprueba que los resultados obtenidos por el hardware coincidan con los valores esperados en la memoria DTCM.
2. **Auditoría del ensamblador:** comprueba la presencia real de instrucciones matriciales Zvt dentro del ejecutable `.elf`.

De esta manera no solamente verificamos que el programa produzca los resultados correctos, sino también que se estén utilizando las instrucciones de la extensión matricial.

---

### 5. Proyecto final P4 — Pipeline de visión de seis etapas

Directorio: `examples/mi_proyecto_npu/`

Como integración final de la investigación, desarrollamos un proyecto que combina las tres unidades de procesamiento de la Coral NPU V2:

- Unidad escalar / FPU.
- Unidad vectorial RVV.
- Unidad matricial VME/Zvt.

El sistema procesa una imagen en escala de grises de dimensiones **128 × 128 píxeles**, utilizando el archivo:

```text
input_128x128.npy
```

#### ¿Por qué utilizamos una imagen de 128 × 128?

La imagen contiene:

\[
128 \times 128 = 16\,384 \text{ píxeles}
\]

Al utilizar un byte por píxel, ocupa:

\[
16\,384 \text{ bytes} = 16 \text{ KB}
\]

Escogimos esta resolución para trabajar dentro de la capacidad de los 32 KB de memoria DTCM disponibles en la arquitectura, considerando también el espacio necesario para los demás búferes y variables del programa.

#### Ejecución del proyecto

Desde el repositorio principal de Coral NPU:

```bash
bazel test --test_output=all //examples/mi_proyecto_npu:testbench_pipeline_img
```

#### Funcionamiento de las seis etapas

El pipeline está compuesto por las siguientes etapas:

**1. Carga de imagen mediante AXI4**

El testbench `testbench_pipeline_img.py` carga los 16 384 píxeles del archivo `input_128x128.npy` directamente en la memoria DTCM del sistema.

**2. Preprocesamiento vectorial (RVV)**

Se realiza el ajuste de brillo y contraste, seguido de una operación de umbralización.

El procesamiento se realiza mediante bloques vectoriales de 16 elementos, utilizando la configuración `e8m1`.

**3. Convolución espacial 2D (VME/Zvt)**

Se aplican filtros de detección de bordes mediante operaciones de multiplicación y acumulación matricial (MAC).

Esta etapa permite aprovechar la unidad matricial de la Coral NPU para realizar operaciones sobre regiones de la imagen.

**4. Activación y normalización escalar (FPU)**

Se calculan estadísticas globales de la imagen, como la media y la varianza, para realizar el proceso de normalización utilizando operaciones de punto flotante FP32.

**5. Postprocesamiento vectorial (RVV)**

Se realiza la saturación de los valores y el empaquetado final de los datos al formato `uint8_t`.

**6. Extracción y validación de resultados**

Una vez finalizada la ejecución del pipeline:

- Python lee los datos almacenados en la DTCM palabra por palabra.
- Se extraen los resultados obtenidos por el hardware.
- Se guardan los datos en `resultados_hw_zvt.npz`.
- Se genera la imagen final `resultado_pipeline_propio.png`.
- Se comparan los resultados del hardware frente a la implementación de referencia en NumPy.

La validación obtiene:

\[
\boxed{MSE = 0.0}
\]

Esto indica que los resultados obtenidos coinciden exactamente con los resultados de referencia de NumPy para las operaciones y los datos evaluados.

---

## 🛠️ Problemas frecuentes y soluciones

### 1. Desbordamiento de la memoria DTCM

La Coral NPU V2 dispone de las siguientes memorias:

| Memoria | Dirección base | Capacidad |
|---|---|---:|
| ITCM | `0x00000000` | 8 KB |
| DTCM | `0x00010000` | 32 KB |

Si aumentamos demasiado el tamaño de los arreglos o variables almacenados en las secciones `.data` y `.bss`, el enlazador `lld` puede indicar que se superó el espacio disponible.

Por esta razón utilizamos imágenes de 128 × 128 píxeles, buscando mantener los datos y búferes dentro de la capacidad física de la DTCM.

### 2. Error `ModuleNotFoundError` en Python

Si Python indica que falta algún módulo, debemos instalar las dependencias necesarias:

```bash
pip install numpy pillow matplotlib cocotb
```

### 3. Verificar si realmente se utilizan instrucciones RVV o Zvt

Para comprobar las instrucciones generadas por el compilador podemos utilizar `llvm-objdump`.

Por ejemplo, para inspeccionar el ejecutable del proyecto final:

```bash
llvm-objdump -d bazel-bin/examples/mi_proyecto_npu/pipeline_img_elf.elf | less
```

También podemos utilizar `llvm-readelf` para inspeccionar la información del ejecutable ELF:

```bash
llvm-readelf -h bazel-bin/examples/mi_proyecto_npu/pipeline_img_elf.elf
```

Estas herramientas permiten analizar el código máquina generado y comprobar qué instrucciones contiene el ejecutable.

---

## 🎯 Objetivo y resultados de la investigación

El propósito de este trabajo fue comprender cómo se distribuye y ejecuta el procesamiento en una arquitectura heterogénea basada en RISC-V, identificando las diferencias entre las unidades escalar, vectorial y matricial.

A lo largo de la investigación pasamos de ejecutar ejemplos oficiales a desarrollar ejercicios propios y finalmente integrar las tres unidades en una aplicación de procesamiento de imágenes.

Los resultados nos permitieron:

- Comprender el funcionamiento de la unidad escalar y la FPU.
- Explorar el uso de intrínsecos y registros de la extensión vectorial RVV.
- Investigar y probar las operaciones matriciales MAC mediante VME/Zvt.
- Analizar el almacenamiento y acceso a datos en las memorias internas.
- Comprobar los resultados mediante simulación RTL con Verilator y Cocotb.
- Inspeccionar los ejecutables para verificar las instrucciones generadas por el compilador.
- Integrar las tres unidades de procesamiento en un pipeline de visión artificial.

**Este repositorio reúne el proceso de investigación, los ejercicios desarrollados, las herramientas utilizadas y la implementación final realizada sobre la Coral NPU V2.**
