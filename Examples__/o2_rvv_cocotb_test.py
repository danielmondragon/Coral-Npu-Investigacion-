import cocotb
import numpy as np
from coralnpu_test_utils.core_mini_axi_interface import CoreMiniAxiInterface
from bazel_tools.tools.python.runfiles import runfiles

@cocotb.test()
async def o2_rvv_add_intrinsic_test(dut):
    core = CoreMiniAxiInterface(dut)
    await core.init()
    await core.reset()
    cocotb.start_soon(core.clock.start())

    r = runfiles.Create()
    elf_path = r.Rlocation("coralnpu_hw/examples/coralnpu_v2_rvv_add_intrinsic.elf")
    with open(elf_path, "rb") as f:
        entry = await core.load_elf(f)
        in1_addr = core.lookup_symbol(f, "input_1")
        in2_addr = core.lookup_symbol(f, "input_2")
        out_addr = core.lookup_symbol(f, "output")

    await core.execute_from(entry)
    await core.wait_for_halted(timeout_cycles=50000)

    in1_bytes = b"".join([(await core.read_word(in1_addr + 4*i)).tobytes() for i in range(256)])
    in2_bytes = b"".join([(await core.read_word(in2_addr + 4*i)).tobytes() for i in range(256)])
    out_bytes = b"".join([(await core.read_word(out_addr + 4*i)).tobytes() for i in range(512)])

    in1 = np.frombuffer(in1_bytes, dtype=np.int8)
    in2 = np.frombuffer(in2_bytes, dtype=np.int8)
    out = np.frombuffer(out_bytes, dtype=np.int16)

    print(f"input_1[0:16] : {in1[:16].tolist()} ... (total={len(in1)} elem int8)")
    print(f"input_2[0:16] : {in2[:16].tolist()} ... (total={len(in2)} elem int8)")
    print(f"output[0:16]  : {out[:16].tolist()} ... (total={len(out)} elem int16, min={out.min()}, max={out.max()})")
    assert np.all(out == 7), f"Error: {out}"
    print("Estado simulacion O2 Oficial RVV Add Intrinsic : Result: PASS")
