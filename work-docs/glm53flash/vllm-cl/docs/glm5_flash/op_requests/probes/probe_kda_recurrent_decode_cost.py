"""归属 `_run_kda` decode 路径（kda_recurrent）里那 ~41 ms/层的开销。

`vllm_cl/patch/patch_kda.py:_run_kda` 在 decode 时每个序列、每层都会做：

    idx = int(ssm_state_indices[si].item())     # (a) host 同步
    h   = initial_state[idx].float().clone()    # (b) 整型索引取页
    ...
    initial_state[idx].copy_(h)                 # (c) 整型索引写页

本脚本在同一台设备上分别计时 (a)/(b)/(c)，判断哪一项是主因。

用法：
    python -u probe_kda_recurrent_decode_cost.py dlc:0 [num_layers]
"""

import sys
import time

import torch

def main() -> int:
    device = sys.argv[1] if len(sys.argv) > 1 else "dlc:0"
    num_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 34
    dev = torch.device(device)

    blocks, heads, dv, dk = 1168, 8, 128, 128
    state = torch.zeros(blocks, heads, dv, dk, dtype=torch.float32, device=dev)
    state.uniform_()
    idx_t = torch.tensor([7], dtype=torch.int32, device=dev)
    idx_long = torch.tensor([7], dtype=torch.int64, device=dev)
    src = torch.zeros(1, heads, dv, dk, dtype=torch.float32, device=dev)

    print(f"device={device} state=[{blocks},{heads},{dv},{dk}] fp32 "
          f"({state.numel() * 4 / 2**20:.0f} MiB) layers={num_layers}")

    def timeit(fn, n=num_layers):
        fn()  # warmup
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n * 1e3

    d2h = timeit(lambda: int(idx_t[0].item()))
    getitem = timeit(lambda: state[7].clone())
    getitem_float = timeit(lambda: state[7].float().clone())
    setitem = timeit(lambda: state[7].copy_(src[0]))
    isel = timeit(lambda: state.index_select(0, idx_long).clone())
    icopy = timeit(lambda: state.index_copy_(0, idx_long, src))
    page = heads * dv * dk * 4

    print(f"  (a) int(idx[0].item())          {d2h:9.3f} ms/call")
    print(f"  (b) state[7].clone()            {getitem:9.3f} ms/call")
    print(f"  (b') state[7].float().clone()   {getitem_float:9.3f} ms/call")
    print(f"  (c) state[7].copy_(page)        {setitem:9.3f} ms/call")
    print(f"  index_select(0,[7]).clone()     {isel:9.3f} ms/call")
    print(f"  index_copy_(0,[7],src)          {icopy:9.3f} ms/call")
    print(f"  1 页 = {page / 2**10:.0f} KiB, 整池 = {state.numel() * 4 / 2**20:.0f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
