load("//rules:coco_tb.bzl", "cocotb_test_suite")
load("//rules:coralnpu_v2.bzl", "coralnpu_v2_binary")

package(default_visibility = ["//visibility:public"])

# Compila el programa principal del pipeline de vision (main.cc) a binario .elf para Coral NPU.
coralnpu_v2_binary(
    name = "pipeline_img_elf",
    srcs = ["main.cc"],
)

# Define la prueba en Verilator + Cocotb que ejecuta el pipeline y genera los resultados.
cocotb_test_suite(
    name = "pipeline_img_hw_test",
    simulators = ["verilator"],
    testcases = ["pipeline_vision_3_motores_test"],
    testcases_vname = "PIPELINE_IMG_TESTCASES",
    tests_kwargs = {
        "waves": False,
        "hdl_toplevel": "VmeCoreMiniAxi",
        "seed": "42",
        "test_module": ["testbench_pipeline_img.py"],
        "deps": [
            "//coralnpu_test_utils:sim_test_fixture",
            "@bazel_tools//tools/python/runfiles",
        ],
        "data": [":pipeline_img_elf"],
        "size": "large",
        "timeout": "eternal",
    },
    verilator_model = "//tests/cocotb:vme_core_mini_axi_model",
)
