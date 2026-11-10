"""设备体检：确认某张 DLC 卡是否还能跑最普通的算子（卡死后必须复位）。

用法：python probe_device_health.py [dlc:0 dlc:1 ...]（默认 0~7）
判据：`exit=0` 且打印 OK = 健康；超时（timeout 124）= 已卡死，需要复位。
"""
import subprocess
import sys

PY = sys.executable
SNIP = (
    "import sys, torch\n"
    "n = int(sys.argv[1])\n"
    "torch.dlc.set_device(n)\n"
    "x = torch.randn(1024, 1024, dtype=torch.bfloat16, device=f'dlc:{n}')\n"
    "y = (x @ x).sum()\n"
    "torch.dlc.synchronize()\n"
    "print(f'dlc:{n} OK sum={y.item():.1f}', flush=True)\n"
)
devs = sys.argv[1:] or [f"dlc:{i}" for i in range(8)]
for d in devs:
    n = d.split(":")[-1]
    try:
        r = subprocess.run([PY, "-c", SNIP, n], capture_output=True, text=True, timeout=120)
        out = "\n".join(l for l in (r.stdout + r.stderr).splitlines()
                        if "DLCSynapse" not in l and "HugePages" not in l)
        print(f"{d}: exit={r.returncode} {out.strip()[:200]}", flush=True)
    except subprocess.TimeoutExpired:
        print(f"{d}: 超时（120s）→ 该卡已卡死，需要复位", flush=True)
