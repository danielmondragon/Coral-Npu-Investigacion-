package(default_visibility = ["//visibility:public"])

# 1. Compila el binario C de su aplicación
cc_binary(
    name = "pipeline_img_elf",
    srcs = [
        "main.c",
        "coralnpu_math.h",
    ],
)

# 2. Define la prueba RTL en Cocotb/Verilator
# Si en su repositorio la regla se llama diferente (ej. coralnpu_cocotb_test),
# usamos la regla estándar de Python test de Bazel:
py_test(
    name = "pipeline_img_test",
    srcs = ["testbench_pipeline_img.py"],
    main = "testbench_pipeline_img.py",
    deps = [
        # Incluye las dependencias de cocotb si son necesarias
    ],
)