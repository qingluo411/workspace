import argparse, os, re

HDR = re.compile(("^(?:─)+ launch  (\\S+)  on dlc:(\\d+) .*nbytes=(\\d+)\\s*$").encode())
FTR = re.compile(("^(?:─)+ cycles: (\\d+)  ops: (\\d+)  bytes: (\\d+)  crt: (\\d+)").encode())
SPLIT_KERNEL = b"custom_dlc_kda_recurrent_decode"

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--offset", type=int, default=0)
ap.add_argument("--out-dir", required=True)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

name = None
end = None
cut = None
pos = a.offset
with open(a.log, "rb") as h:
    h.seek(a.offset)
    for line in h:
        m = HDR.match(line)
        if m:
            if name is not None:
                end = pos
            name = m.group(1)
            if cut is None and name == SPLIT_KERNEL:
                cut = pos
        elif name is not None and FTR.match(line):
            end = pos + len(line)
            name = None
        pos += len(line)

end = pos if end is None else end
assert cut, "split kernel not found"
print(f"window={end - a.offset:,}B  prefill={cut - a.offset:,}B  decode={end - cut:,}B")

with open(a.log, "rb") as h:
    h.seek(a.offset)
    with open(os.path.join(a.out_dir, "prefill.ansi"), "wb") as f:
        f.write(h.read(cut - a.offset))
    with open(os.path.join(a.out_dir, "decode.ansi"), "wb") as f:
        f.write(h.read(end - cut))
