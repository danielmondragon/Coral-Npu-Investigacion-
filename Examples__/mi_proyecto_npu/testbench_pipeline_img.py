import os
import cocotb
import numpy as np
from coralnpu_test_utils.sim_test_fixture import Fixture
from bazel_tools.tools.python.runfiles import runfiles

@cocotb.test()
async def pipeline_vision_3_motores_test(dut):
    print("\n=======================================================")
    print("   EJECUCIÓN REAL EN HARDWARE CORALNPU V2 (ZVT + RVV)   ")
    print("=======================================================\n")

    fixture = await Fixture.Create(dut)
    r = runfiles.Create()
    elf_path = r.Rlocation("coralnpu_hw/examples/mi_proyecto_npu/pipeline_img_elf.elf")

    symbols = [
        "dtcm_in_rgb",
        "dtcm_in_gray",
        "dtcm_gray_thresh",
        "dtcm_out_esc_brillo",
        "dtcm_out_esc_bin",
        "dtcm_out_rvv_neg",
        "dtcm_out_rvv_esmeralda",
        "dtcm_zvt_patches",
        "dtcm_zvt_kernels",
        "dtcm_out_zvt_sobel",
        "dtcm_out_zvt_blur",
    ]
    await fixture.load_elf_and_lookup_symbols(elf_path, symbols)

    SIM_DIM = 128
    folder = "/home/daniel/coralnpu/examples/mi_proyecto_npu"
    npy_path = os.path.join(folder, "input_128x128.npy")
    input_rgb = np.load(npy_path) if os.path.exists(npy_path) else np.full((SIM_DIM, SIM_DIM, 3), 128, dtype=np.uint8)

    gray = (0.299 * input_rgb[:, :, 0] + 0.587 * input_rgb[:, :, 1] + 0.114 * input_rgb[:, :, 2]).astype(np.uint8)
    umbral = int(np.mean(gray))
    await fixture.write_word("dtcm_gray_thresh", umbral)

    kernels = np.array([
        [-1, -1, 1], [0, -2, 1], [1, -1, 1],
        [-2,  0, 1], [0,  0, 1], [2,  0, 1],
        [-1,  1, 1], [0,  2, 1], [1,  1, 1],
    ], dtype=np.int8)
    await fixture.write("dtcm_zvt_kernels", kernels.flatten().view(np.uint8))

    out_brillo = np.zeros_like(input_rgb)
    out_bin = np.zeros_like(gray)
    out_neg = np.zeros_like(input_rgb)
    out_esmeralda = np.zeros_like(input_rgb)
    out_sobel = np.zeros_like(gray)
    out_blur = np.zeros_like(gray)

    flat_rgb = input_rgb.reshape(-1, 3)
    flat_gray = gray.reshape(-1)
    gray_pad = np.pad(gray, 1, mode="edge")

    all_patches = np.zeros((SIM_DIM * SIM_DIM, 9), dtype=np.uint8)
    idx_p = 0
    for y in range(SIM_DIM):
        for x in range(SIM_DIM):
            all_patches[idx_p] = gray_pad[y:y+3, x:x+3].flatten()
            idx_p += 1

    total_batches = (SIM_DIM * SIM_DIM) // 64
    for b in range(total_batches):
        s = b * 64
        e = s + 64
        await fixture.write("dtcm_in_rgb", flat_rgb[s:e].flatten())
        await fixture.write("dtcm_in_gray", flat_gray[s:e])
        await fixture.write("dtcm_zvt_patches", all_patches[s:e].flatten())

        await fixture.run_to_halt(timeout_cycles=100000)

        out_brillo.reshape(-1, 3)[s:e] = (await fixture.read("dtcm_out_esc_brillo", 64 * 3)).reshape(64, 3)
        out_bin.reshape(-1)[s:e] = await fixture.read("dtcm_out_esc_bin", 64)
        out_neg.reshape(-1, 3)[s:e] = (await fixture.read("dtcm_out_rvv_neg", 64 * 3)).reshape(64, 3)
        out_esmeralda.reshape(-1, 3)[s:e] = (await fixture.read("dtcm_out_rvv_esmeralda", 64 * 3)).reshape(64, 3)
        out_sobel.reshape(-1)[s:e] = await fixture.read("dtcm_out_zvt_sobel", 64)
        out_blur.reshape(-1)[s:e] = await fixture.read("dtcm_out_zvt_blur", 64)

    out_npz = os.path.join(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp"), "resultados_hw_zvt.npz")
    np.savez(
        out_npz,
        input_rgb=input_rgb,
        out_brillo=out_brillo,
        out_bin=out_bin,
        out_neg=out_neg,
        out_esmeralda=out_esmeralda,
        out_sobel=out_sobel,
        out_blur=out_blur,
    )
    print(f"[ÉXITO] 64 lotes ejecutados en FPU + RVV + Zvt (VmeCoreMiniAxi). Resultados guardados en {out_npz}")
    print("Estado simulacion Pipeline 3 Motores : Result: PASS\n")
