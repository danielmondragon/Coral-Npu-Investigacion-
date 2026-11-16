Coral-Npu-Investigacion-/
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
