#!/usr/bin/env python3
"""CUDA warmup + sanity check.

Runs a small GEMM and a graph-conv op so cuDNN/cuBLAS/PyG kernels JIT-compile
and the GPU clocks reach steady state before the long-running training kicks
off. Also reports the device, driver, free VRAM, and a microbenchmark TFLOPS
number so you have a quick health signal for the GPU.
"""

import time

import torch


def fmt_bytes(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} PB"


def main():
    print("=" * 60)
    print("CUDA WARMUP + SANITY CHECK")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA NOT AVAILABLE — training will run on CPU.")
        return

    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    print(f"Device          : {props.name}")
    print(f"Compute         : sm_{props.major}{props.minor}")
    print(f"VRAM total      : {fmt_bytes(props.total_memory)}")
    print(f"SMs             : {props.multi_processor_count}")
    print(f"PyTorch         : {torch.__version__}")
    print(f"CUDA runtime    : {torch.version.cuda}")
    print(f"cuDNN           : {torch.backends.cudnn.version()}")

    free, total = torch.cuda.mem_get_info()
    print(f"VRAM free       : {fmt_bytes(free)} / {fmt_bytes(total)}")

    # 1) GEMM warmup — matmul + cuBLAS init
    print("\n[1/3] GEMM warmup (4096x4096 fp32, 5 iters)…")
    a = torch.randn(4096, 4096, device=dev)
    b = torch.randn(4096, 4096, device=dev)
    torch.cuda.synchronize()
    for _ in range(2):  # discard first 2 (kernel autotune)
        _ = a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 5
    flops = 2 * 4096 ** 3
    tflops = flops / dt / 1e12
    print(f"      one matmul: {dt * 1000:.2f} ms  ({tflops:.1f} TFLOPS fp32)")

    # 2) Conv warmup — exercises cuDNN
    print("\n[2/3] Conv warmup (cuDNN init)…")
    x = torch.randn(8, 64, 64, 64, device=dev)
    w = torch.randn(64, 64, 3, 3, device=dev)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = torch.nn.functional.conv2d(x, w, padding=1)
    torch.cuda.synchronize()
    print(f"      conv2d 8x64x64x64: {(time.perf_counter() - t0) * 1000:.2f} ms")

    # 3) Graph op warmup — torch_scatter / torch_sparse if installed, else PyG SAGEConv
    print("\n[3/3] Graph op warmup (SAGEConv 50k nodes / 200k edges)…")
    try:
        from torch_geometric.nn import SAGEConv
        n_nodes, n_edges, in_dim, out_dim = 50_000, 200_000, 64, 32
        feat = torch.randn(n_nodes, in_dim, device=dev)
        edge_idx = torch.randint(0, n_nodes, (2, n_edges), device=dev)
        sage = SAGEConv(in_dim, out_dim).to(dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = sage(feat, edge_idx)
        torch.cuda.synchronize()
        print(f"      SAGEConv forward: {(time.perf_counter() - t0) * 1000:.2f} ms")
    except ImportError as e:
        print(f"      torch_geometric not available: {e}")

    free_after, _ = torch.cuda.mem_get_info()
    print(f"\nVRAM free after warmup: {fmt_bytes(free_after)} (used "
          f"{fmt_bytes(free - free_after)} during warmup)")

    # Cleanup
    del a, b, c, x, w
    if "feat" in dir():
        del feat, edge_idx, sage
    torch.cuda.empty_cache()
    print("\nWarmup complete. GPU is ready for training.")


if __name__ == "__main__":
    main()
