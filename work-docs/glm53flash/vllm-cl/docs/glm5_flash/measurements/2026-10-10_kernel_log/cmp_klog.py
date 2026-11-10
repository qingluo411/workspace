import re, sys, collections
def load(p):
    d={}
    for line in open(p):
        m=re.match(r"\s+(\d+)\s+([\d,]+)\s+([\d.]+)%\s+(\S+)$", line)
        if m: d[m.group(4)]=(int(m.group(1)), int(m.group(2).replace(",","")))
    return d
old=load(sys.argv[1]); new=load(sys.argv[2])
to=sum(v[1] for v in old.values()); tn=sum(v[1] for v in new.values())
print(f"TOTAL old={to:,}  new={tn:,}  ratio={to/tn:.2f}x")
print(f"{'operator':<70} {'old_calls':>9} {'new_calls':>9} {'old_cyc':>14} {'new_cyc':>14} {'old%':>6} {'new%':>6} {'d%':>7}")
names=sorted(set(old)|set(new), key=lambda n:-(old.get(n,(0,0))[1]+new.get(n,(0,0))[1]))
for n in names:
    oc,ocy=old.get(n,(0,0)); nc,ncy=new.get(n,(0,0))
    op=100.0*ocy/to; np_=100.0*ncy/tn
    print(f"{n[:70]:<70} {oc:>9} {nc:>9} {ocy:>14,} {ncy:>14,} {op:>6.2f} {np_:>6.2f} {np_-op:>+7.2f}")
