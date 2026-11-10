import torch
dev='dlc'
def run(desc, kpe_w, block_size=256, nb=4):
    try:
        kv = torch.zeros(nb, block_size, 512+kpe_w, dtype=torch.bfloat16, device=dev)
        T = 6
        kvc = torch.randn(T, 512, dtype=torch.bfloat16, device=dev)
        kpe = torch.randn(T, kpe_w, dtype=torch.bfloat16, device=dev)
        slot = (torch.arange(T, dtype=torch.int64, device=dev) * 7 + 3) % (nb*block_size)
        torch.dlc_vllm_concat_and_cache_mla_(kv, kvc, kpe, slot)
        torch.dlc.synchronize()
        got = kv.reshape(-1, 512+kpe_w)[slot]
        err = (got[:, :512].float() - kvc.float()).abs().max().item()
        err_pe = (got[:, 512:].float() - kpe.float()).abs().max().item() if kpe_w else 0.0
        print(f"[{desc}] OK  latent_err={err:.3e} kpe_err={err_pe:.3e}")
    except Exception as e:
        print(f"[{desc}] FAILED: {type(e).__name__}: {str(e)[:400]}")
run("kpe=64  576-layout bs=256", 64)
run("kpe=0   512-NoPE   bs=256", 0)
