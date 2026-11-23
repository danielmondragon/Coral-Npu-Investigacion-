import cocotb
import numpy as np
from coralnpu_test_utils.sim_test_fixture import Fixture
from bazel_tools.tools.python.runfiles import runfiles

@cocotb.test()
async def zvt_matrix_mac_dot_test(dut):
    # Creamos la prueba en Verilator y cargamos el binario .elf junto con sus simbolos en DTCM.
    fixture = await Fixture.Create(dut)
    r = runfiles.Create()
    elf_path = r.Rlocation("coralnpu_hw/examples/coralnpu_v2_matrix_mac_dot.elf")
    symbols = ["weight_matrix", "feature_vector", "matmul_4x4_out", "acc_output"]
    await fixture.load_elf_and_lookup_symbols(elf_path, symbols)

    # Corremos el programa en el hardware simulado hasta que se detenga.
    await fixture.run_to_halt()

    # Leemos las matrices de entrada y las salidas calculadas por Zvt desde DTCM.
    w = (await fixture.read("weight_matrix", 16)).view(np.int8).reshape(4, 4)
    f = (await fixture.read("feature_vector", 16)).view(np.uint8).reshape(4, 4)
    c_hw = (await fixture.read("matmul_4x4_out", 16 * 4)).view(np.int32).reshape(4, 4)
    diag_hw = (await fixture.read("acc_output", 4 * 4)).view(np.int32)

    # Comparamos el resultado del hardware contra la multiplicacion matricial esperada.
    c_expected = w.astype(np.int32) @ f.astype(np.int32)
    assert np.array_equal(c_hw, c_expected), f"Mismatch:\n{c_hw}\nvs\n{c_expected}"

    # Imprimimos las matrices y la diagonal acumulada para revisar en consola.
    print("\n--- MATRIZ DE PESOS A (4x4 int8 en DTCM) ---")
    print(w)
    print("\n--- MATRIZ DE ENTRADA B (4x4 uint8 en DTCM) ---")
    print(f)
    print("\n--- PRODUCTO MATRICIAL C = A x B (Hardware Zvt VmeCoreMiniAxi) ---")
    print(c_hw)
    print("\n--- DIAGONAL ACUMULADA (acc_output) ---")
    print(diag_hw.tolist())
    print("Estado simulacion Zvt : Result: PASS\n")
