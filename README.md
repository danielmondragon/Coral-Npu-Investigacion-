# Investigación y Aplicación en Coral NPU (RISC-V Vectorial)

Este repositorio contiene la documentación, los scripts y los ejemplos desarrollados para la exploración, simulación e integración de algoritmos sobre el coprocesador vectorial y la arquitectura de la **Coral NPU** (RISC-V RV32 con extensiones vectoriales VME/Zvt).

## 👥 Integrantes
* Alejandra Calderon Zambrana
* Daniel Mondragón

## 📋 Estructura y Contenido del Repositorio

El proyecto se encuentra organizado de la siguiente manera:

```text
Coral-Npu-Investigacion/
├── README.md                              ← este archivo
├── docs/                                  ← 📄 LOS 3 PDF (leer en orden 1 → 2 → 3)
│   ├── 01_Guia_Instalacion_CoralNPU.pdf
│   ├── 02_Informe_Ejemplos_CoralNPU.pdf
│   └── 03_Documentacion_Completa_CoralNPU.pdf
└── examples/                              ← se copia dentro de ~/coralnpu/examples
    ├── BUILD.bazel                        ← reglas Bazel: 2 ejemplos oficiales + 3 propios
    ├── BUILD                              ← versión reducida (solo scalar_bias_relu)
    ├── coralnpu_math.h                    ← librería propia (tipos y operaciones MAC)
    ├── hello_world_add_floats.cc          ← [oficial] suma escalar de flotantes
    ├── rvv_add_intrinsic.cc               ← [oficial] suma vectorial con intrínsecos RVV
    ├── coralnpu_v2_scalar_bias_relu.cc    ← [propio] escalar: bias + ReLU
    ├── coralnpu_v2_rvv_quantized_scale.cc ← [propio] vectorial: escalado cuantizado
    └── mi_proyecto_npu/                   ← [propio] proyecto integrador de imágenes
        ├── BUILD
        ├── main.c                         ← orquesta las 3 etapas (escalar, SIMD, MAC)
        ├── coralnpu_math.h                ← funciones de filtros de imagen
        ├── testbench_pipeline_img.py      ← testbench Python: carga la foto y genera el dashboard
        ├── aaaaaaa.jpg                    ← imagen de entrada
        └── resultado_pipeline_propio.png  ← resultado: 6 filtros aplicados
