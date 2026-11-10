import torch

PAD_SLOT_ID = -1

def kda_state_gather_ref(state: torch.Tensor, indices: torch.Tensor,
                         has_initial_state: torch.Tensor,
                         out: torch.Tensor | None = None) -> torch.Tensor:
    """state [P,H,Dv,Dk] (any stride(0)) -> out [S,H,Dv,Dk]"""
    P = state.shape[0]
    S = indices.numel()
    shape = (S,) + tuple(state.shape[1:])
    if out is None:
        out = torch.zeros(shape, dtype=state.dtype, device=state.device)
    else:
        out.zero_()
    for s in range(S):
        p = int(indices[s])
        if p == PAD_SLOT_ID:
            continue
        if not (0 <= p < P):
            raise IndexError(f"gather page index out of range: {p}")
        if bool(has_initial_state[s]):
            out[s].copy_(state[p])
    return out

def kda_state_scatter_ref(state: torch.Tensor, src: torch.Tensor,
                          indices: torch.Tensor) -> None:
    """state [P,H,Dv,Dk] in-place (any stride(0)); indices<0 -> skip"""
    P = state.shape[0]
    if src.shape[0] != indices.numel() or tuple(src.shape[1:]) != tuple(state.shape[1:]):
        raise ValueError("scatter shape mismatch")
    seen = set()
    for s in range(indices.numel()):
        p = int(indices[s])
        if p == PAD_SLOT_ID:
            continue
        if not (0 <= p < P):
            raise IndexError(f"scatter page index out of range: {p}")
        if p in seen:
            raise ValueError("duplicate writable KDA cache pages")
        seen.add(p)
        state[p].copy_(src[s])

if __name__ == "__main__":
    torch.manual_seed(0)
    P, H, Dv, Dk = 7, 2, 4, 5
    page = H * Dv * Dk
    # padded pool: page stride = page + 3 elements
    flat = torch.arange(P * (page + 3), dtype=torch.float32)
    state = flat[3:].view(P, H, Dv, Dk) if False else flat.as_strided((P, H, Dv, Dk), (page + 3, Dv * Dk, Dk, 1))
    idx = torch.tensor([3, PAD_SLOT_ID, 0, 5, 5])
    has = torch.tensor([True, True, False, True, True])
    out = kda_state_gather_ref(state, idx, has)
    assert torch.equal(out[1], torch.zeros((H, Dv, Dk)))          # pad slot -> zero
    assert torch.equal(out[2], torch.zeros((H, Dv, Dk)))          # no initial state -> zero
    assert torch.equal(out[0], state[3]) and torch.equal(out[4], state[5])
    src = torch.full((idx.numel(), H, Dv, Dk), 9.0)
    before = state.clone()
    sidx = torch.tensor([3, PAD_SLOT_ID, 0, 5, 6])          # scatter 不允许重复页
    kda_state_scatter_ref(state, src, sidx)
    assert torch.equal(state[3], src[0]) and torch.equal(state[0], src[2])
    assert torch.equal(state[5], src[3]) and torch.equal(state[6], src[4])
    assert torch.equal(state[1], before[1]) and torch.equal(state[4], before[4])   # untouched pages
    try:
        kda_state_scatter_ref(state, src, idx)              # idx 含重复页 5,5
        raise AssertionError("duplicate scatter page must raise")
    except ValueError:
        pass
    print("reference self-test OK  (strided state, stride0=%d, page=%d)" % (state.stride(0), page))
