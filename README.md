# Investigación y Aplicación en Coral NPU (RISC-V Vectorial)

En este repositorio guardamos toda la documentación, los códigos y los ejemplos que hicimos para probar y entender cómo funciona por dentro el coprocesador vectorial y la arquitectura de la **Coral NPU** (RISC-V RV32 con extensiones vectoriales VME/Zvt).

## 👥 Integrantes
* Alejandra Calderon Zambrana
* Daniel Mondragón

## 📋 Cómo organizamos nuestros archivos

Fuimos ordenando todo el trabajo de la siguiente manera:

    Coral-Npu-Investigacion/
    ├── README.md                              ← este archivo
    ├── docs/                                  ← 📄 LOS 3 PDF (les recomendamos leerlos en orden 1 → 2 → 3)
    │   ├── 01_Guia_Instalacion_CoralNPU.pdf
    │   ├── 02_Informe_Ejemplos_CoralNPU.pdf
    │   └── 03_Documentacion_Completa_CoralNPU.pdf
    └── examples/                              ← carpeta que llevamos a ~/coralnpu/examples
        ├── BUILD.bazel                        ← reglas Bazel completas: los 2 oficiales + nuestros 3 códigos
        ├── BUILD                              ← versión corta que usamos al inicio (solo scalar_bias_relu)
        ├── coralnpu_math.h                    ← nuestra librería base (tipos y operaciones MAC)
        ├── hello_world_add_floats.cc          ← [oficial] suma escalar de flotantes
        ├── rvv_add_intrinsic.cc               ← [oficial] suma vectorial con intrínsecos RVV
        ├── coralnpu_v2_scalar_bias_relu.cc    ← [propio] prueba escalar: bias + ReLU
        ├── coralnpu_v2_rvv_quantized_scale.cc ← [propio] prueba vectorial: escalado cuantizado
        └── mi_proyecto_npu/                   ← [propio] nuestro proyecto final con imágenes
            ├── BUILD
            ├── main.c                         ← junta las 3 etapas (escalar, SIMD, MAC)
            ├── coralnpu_math.h                ← funciones de nuestros filtros
            ├── testbench_pipeline_img.py      ← script en Python: carga la foto y arma la imagen final
            ├── aaaaaaa.jpg                    ← foto de prueba que usamos
            └── resultado_pipeline_propio.png  ← resultado con los 6 filtros aplicados

---

## 📚 En qué orden revisar nuestros PDF (`docs/`)

Para que se entienda bien todo lo que hicimos desde el día uno hasta que logramos procesar fotos completas, dejamos los tres informes en `docs/` pensados para leerse en orden:

1. **`01_Guia_Instalacion_CoralNPU.pdf`**
   * **Lo que pusimos aquí:** Anotamos todo el paso a paso de cómo instalamos y dejamos listo el entorno en Linux / WSL2, el compilador de RISC-V, Bazel y el simulador Verilator / Cocotb.
   * **Para tener en cuenta:** Si van a probar nuestros códigos, asegúrense de tener la carpeta del proyecto en `~/coralnpu` con las mismas rutas que dejamos en esta guía.
2. **`02_Informe_Ejemplos_CoralNPU.pdf`**
   * **Lo que pusimos aquí:** Explicamos con nuestras palabras cómo trabajan las tres partes principales de la NPU (Escalar, Vectorial y Matricial). También mostramos cómo guardamos los datos en la memoria DTCM, comentamos nuestros códigos y pegamos los pantallazos de las pruebas pasando en Verilator.
   * **Para tener en cuenta:** Fíjense en cómo metimos los arreglos dentro de la sección `.data` (DTCM) y en las cuentas que hicimos para comprobar que cada ejercicio (`scalar_bias_relu`, `rvv_quantized_scale` y el filtro de fotos) diera el resultado exacto.
3. **`03_Documentacion_Completa_CoralNPU.pdf`**
   * **Lo que pusimos aquí:** Juntamos toda la investigación en un documento general explicando cómo es el hardware de la CoralNPU V2 por dentro, cómo viajan las instrucciones y nuestras conclusiones.
   * **Para tener en cuenta:** Revisen la tabla comparativa del final, donde mostramos la diferencia real entre hacer las operaciones de forma escalar, vectorial o matricial.

---

## ⚙️ Qué hacer antes de correr los códigos

Para probar nuestros ejemplos en su computador, primero hay que copiar lo que dejamos en `examples/` dentro de la carpeta `examples/` del repositorio principal de Coral NPU:

    cp -r examples/* ~/coralnpu/examples/
    cd ~/coralnpu

> **⚠️ Ojo con los archivos `BUILD`:** En nuestra carpeta `examples/` dejamos dos archivos: `BUILD.bazel` (que tiene las reglas para correr todos los ejemplos) y `BUILD` (una versión corta que usamos al principio solo para probar `scalar_bias_relu`). Como Bazel siempre lee primero el archivo que se llama `BUILD` sin extensión cuando ve los dos juntos, para compilar todo completo hay que borrar o mover `examples/BUILD` y dejar solo `examples/BUILD.bazel`.

---

## 🚀 Cómo probar cada ejercicio

### 1. Ejemplos Oficiales Base (`hello_world_add_floats` y `rvv_add_intrinsic`)
Empezamos compilando estos dos ejemplos que ya traía el repositorio para estar seguros de que el compilador RISC-V y Bazel nos estaban funcionando bien:

    bazel build //examples:hello_world_add_floats
    bazel build //examples:rvv_add_intrinsic

* **¿Qué revisar?:** Que Bazel termine sin errores (`Build completed successfully`) y genere los archivos (`.elf`, `.bin`, `.vmem`) dentro de `bazel-bin/examples/`.

---

### 2. Nuestro Ejercicio 1 — Parte Escalar (`coralnpu_v2_scalar_bias_relu.cc`)
En esta primera prueba propia quisimos ver cómo trabaja la parte escalar con números decimales: agarramos un arreglo de 8 números flotantes, les sumamos un sesgo ($b = 0.5$) y les aplicamos $\text{ReLU}(x) = \max(0, x + b)$ para dejar en cero los negativos.

* **Para compilarlo:**

      bazel build //examples:coralnpu_v2_scalar_bias_relu

* **Para correr la simulación en Verilator:**

      bazel test --test_output=all //examples:verilator_uvm_regression_coralnpu_v2_scalar_bias_relu

* **Lo que hay que mirar:**
  * En `coralnpu_v2_scalar_bias_relu.cc` usamos `__attribute__((section(".data")))` para guardar nuestros datos directo en la memoria **DTCM**.
  * En la consola de Verilator se debe ver cómo el vector de entrada `[-2.5, 1.0, -0.5, 3.0, -4.0, 2.0, 0.0, -1.2]` sale transformado exactamente en `[0.0, 1.5, 0.0, 3.5, 0.0, 2.5, 0.5, 0.0]` y con estado **`PASSED`**.

---

### 3. Nuestro Ejercicio 2 — Parte Vectorial (`coralnpu_v2_rvv_quantized_scale.cc`)
Aquí quisimos probar las instrucciones vectoriales de RISC-V (RVV) haciendo varias operaciones al mismo tiempo. Trabajamos con bloques de números de 8 bits (`int8_t`), pasándolos temporalmente a 16 bits (`int16_t`) mientras se multiplican para no perder datos.

* **Para compilarlo:**

      bazel build //examples:coralnpu_v2_rvv_quantized_scale

* **Para correr la simulación en Verilator:**

      bazel test --test_output=all //examples:verilator_uvm_regression_coralnpu_v2_rvv_quantized_scale

* **Lo que hay que mirar:**
  * Nos aseguramos de que los números no se desbordaran (*overflow*) al multiplicarse antes de recortar el resultado final.
  * Comprobar que la simulación termine en **`PASSED`**.

---

### 4. Nuestro Proyecto Final — Procesamiento de Fotos (`examples/mi_proyecto_npu/`)
Para cerrar el trabajo quisimos hacer algo mucho más visual que solo listas de números. Armamos un programa que pone a trabajar las 3 partes de la NPU (Escalar, Vectorial y Matricial) sobre una foto real de **$512 \times 512$ píxeles a color** (`aaaaaaa.jpg`).

* **Compilación del código C de la NPU:**

      bazel build //examples/mi_proyecto_npu:pipeline_img_elf

* **Ejecución del script en Python y creación de la imagen final:**

      python3 examples/mi_proyecto_npu/testbench_pipeline_img.py

#### ¿Qué hicimos aquí y cómo revisarlo?
1. **Código en C y Memoria DTCM (`coralnpu_math.h` y `main.c`):**
   * Guardamos los espacios de memoria de entrada y salida dentro de la **DTCM** para que los datos viajen rápido.
   * **Parte Escalar (píxel por píxel):**
     * *Brillo y Contraste RGB:* Le aplicamos a cada píxel la cuenta $1.25x + 20$ (sin pasarnos de 255) para que la foto se vea más viva e iluminada.
     * *Blanco y Negro Puro:* Pasamos la foto a grises y comparamos cada punto contra el brillo promedio; los más claros quedan blancos ($255$) y los más oscuros negros ($0$).
   * **Parte Vectorial SIMD (varios colores a la vez):**
     * *Negativo RGB:* Tomamos los 3 colores de cada píxel $[R, G, B]$ juntos y les restamos $255$ al tiempo para invertir todos los colores.
     * *Filtro Esmeralda:* Le bajamos al rojo ($R - 50$) y le subimos al verde ($G + 40$) en paralelo para darle un tono frío y resaltar las zonas verdes.
   * **Parte Matricial MAC (filtros por cuadros 2D):**
     * *Bordes Sobel HD:* Pasamos unas matrices de $3 \times 3$ por toda la foto para encontrar dónde cambian brusco los tonos, dibujando los bordes del edificio en un mapa de calor.
     * *Modo Retrato Bokeh:* Quisimos imitar el efecto de desenfoque de fondo de las cámaras profesionales. Primero desenfocamos toda la foto con una matriz de $15 \times 15$ y luego la mezclamos con la foto original usando una máscara circular centrada en la torre. Así la torre queda bien nítida y todo alrededor se ve borroso.
2. **La imagen resultado (`resultado_pipeline_propio.png`):**
   * Al correr el script se guarda `examples/mi_proyecto_npu/resultado_pipeline_propio.png`.
   * Ahí salen los **6 filtros organizados en una tabla de $2 \times 3$** sobre fondo oscuro, limpios y sin bordes cortados.
   * Si quieren probar nuestro código con cualquier otra foto, solo pongan la ruta al final del comando:

         python3 examples/mi_proyecto_npu/testbench_pipeline_img.py ruta/a/otra_foto.jpg

3. **Librerías de Python:** Para que el script arme la imagen final hay que tener instalados `numpy`, `pillow` y `matplotlib`.

---

## 🛠️ Problemas que nos salieron y cómo arreglarlos rápido

* **Si Bazel dice `Rule 'coralnpu_v2_...' not found`:** Es porque quedó puesto el archivo `examples/BUILD` corto. Bórrenlo para que Bazel use `examples/BUILD.bazel`.
* **Si Python dice `ModuleNotFoundError`:** Instalen las librerías que faltan en WSL2/Linux con `pip install numpy pillow matplotlib`.
* **Si prueban con una foto de otro tamaño:** No hay problema; el código ajusta automáticamente cualquier foto a $512 \times 512$ píxeles antes de procesarla para que nunca dé error de tamaño en las matrices.
