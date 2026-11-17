import cocotb
import numpy as np
from coralnpu_test_utils.core_mini_axi_interface import CoreMiniAxiInterface
from bazel_tools.tools.python.runfiles import runfiles

@cocotb.test()
async def p1_scalar_bias_relu_test(dut):
    core = CoreMiniAxiInterface(dut)
    await core.init()
    await core.reset()
    cocotb.start_soon(core.clock.start())

    r = runfiles.Create()
    elf_path = r.Rlocation("coralnpu_hw/examples/coralnpu_v2_scalar_bias_relu.elf")
    with open(elf_path, "rb") as f:
        entry = await core.load_elf(f)
        in_addr = core.lookup_symbol(f, "input_data")
        bias_addr = core.lookup_symbol(f, "bias_val")
        out_addr = core.lookup_symbol(f, "output_data")

    in_vals = [float(np.frombuffer((await core.read_word(in_addr + 4*i)).tobytes(), dtype=np.float32)[0]) for i in range(8)]
    bias = float(np.frombuffer((await core.read_word(bias_addr)).tobytes(), dtype=np.float32)[0])
    print(f"input_data  : {[round(x, 2) for x in in_vals]}")
    print(f"bias_val    : {round(bias, 2)}")

    await core.execute_from(entry)
    await core.wait_for_halted()

    out_vals = [float(np.frombuffer((await core.read_word(out_addr + 4*i)).tobytes(), dtype=np.float32)[0]) for i in range(8)]
    print(f"output_data : {[round(x, 2) for x in out_vals]}")
    expected = np.maximum(0.0, np.array(in_vals, dtype=np.float32) + bias)
    assert np.allclose(out_vals, expected), f"Error: {out_vals} != {expected}"
    print("Estado simulacion P1 Escalar Bias+ReLU : Result: PASS")
