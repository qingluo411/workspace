import torch
dev='dlc'; torch.manual_seed(0)
T, HC, HS = 64, 4, 256
HC2, HC3 = HC*HC, 2*HC+HC*HC
residual = torch.randn(T, HC, HS, dtype=torch.bfloat16, device=dev)*0.5
fn      = torch.randn(HC3, HC*HS, dtype=torch.float32, device=dev)*0.05
hc_scale= torch.randn(3, dtype=torch.float32, device=dev)*0.1
hc_base = torch.randn(HC3, dtype=torch.float32, device=dev)*0.1
rms_eps, pre_eps, sink_eps, post_mult, repeat = 1e-6, 1e-6, 1e-6, 1.0, 1

post, cm, layer_input = torch.dlc_mhc_pre(residual, fn, hc_scale, hc_base,
                                          rms_eps, pre_eps, sink_eps, post_mult, repeat, 1)
torch.dlc.synchronize()

# reference (same math as the op's CPU fallback / vLLM's mhc_pre reference)
n = T
x = residual.reshape(n, HC*HS).float()
g = x @ fn.float().t()
rms = torch.rsqrt((x*x).sum(-1)/ (HC*HS) + rms_eps)
mixes = g * rms.unsqueeze(-1)
post_ref = (torch.sigmoid(mixes[:, HC:2*HC]*hc_scale[1] + hc_base[HC:2*HC]) * post_mult).reshape(n, HC)
cm_ref = mixes[:, 2*HC:HC3]*hc_scale[2] + hc_base[2*HC:HC3]
cm_ref = cm_ref.reshape(n, HC, HC)
cm_ref = torch.softmax(cm_ref, -1) + sink_eps
cm_ref = cm_ref / (cm_ref.sum(-2, keepdim=True) + sink_eps)
pre = (torch.sigmoid(mixes[:, 0:HC]*hc_scale[0] + hc_base[0:HC]) + pre_eps).reshape(n, HC, 1)
li_ref = (pre * residual.reshape(n, HC, HS).float()).sum(1)

print("mhc_pre  post_err=%.3e cm_err=%.3e layer_input_err=%.3e" % (
    (post.float()-post_ref).abs().max().item(),
    (cm.reshape(n,HC,HC).float()-cm_ref).abs().max().item(),
    (layer_input.float()-li_ref).abs().max().item()),
    "| dtypes:", post.dtype, cm.dtype, layer_input.dtype)

# ---- mhc_post ----
x_tok = torch.randn(T, HS, dtype=torch.bfloat16, device=dev)*0.5
out = torch.dlc_mhc_post(x_tok, residual, post, cm)
torch.dlc.synchronize()
ref = (post.reshape(T, HC, 1)*x_tok.reshape(T,1,HS).expand(T,HC,HS).float()
       + torch.bmm(cm.reshape(T,HC,HC).transpose(1,2), residual.reshape(T,HC,HS).float()))
print("mhc_post out_err=%.3e out_dtype=%s" % ((out.float()-ref).abs().max().item(), out.dtype))
