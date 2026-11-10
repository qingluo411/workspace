# Kimi-K3 INT4 bring-up probes

These probes exercise DLC components without loading the 1.4 TiB checkpoint.

Environment:

```bash
export LD_LIBRARY_PATH=/usr/local/chipltech/dlccl/lib:/usr/local/chipltech/simulator/lib:/usr/local/chipltech/synapse/lib:/usr/local/chipltech/kernel:/usr/local/chipltech/bluejay/build
export CUSTOM_KERNEL_PATH=/usr/local/chipltech/synapse/lib
export DLC_TPU_VERSION=2
```

KDA geometry probe:

```bash
timeout 75 /root/workspace/.venv/bin/python \
  /root/workspace/work-tests/kimi_k3/probe_kda_shapes.py \
  --device dlc:5 --mode decode --heads 96 --tokens 1

timeout 75 /root/workspace/.venv/bin/python \
  /root/workspace/work-tests/kimi_k3/probe_kda_shapes.py \
  --device dlc:5 --mode chunk --heads 96 --tokens 16 --chunk-size 16
```

Always use an external timeout. A DLC kernel-shape failure may leave the queue
blocked until the owning process exits.

Checkpoint tools:

```bash
/root/workspace/.venv/bin/python \
  /root/workspace/work-tests/kimi_k3/audit_checkpoint.py \
  --model /data/models/Kimi-K3-INT4-Slice8-TP8 \
  --output /root/workspace/work-tests/kimi_k3/reports/slice8_tp8_audit_20261028.json
```

The 24-layer TP8 slice is generated with:

```bash
/root/workspace/.venv/bin/python \
  /root/workspace/work-tests/kimi_k3/tools/make_slice_checkpoint.py \
  --source /data/models/Kimi-K3-INT4 \
  --output /data/models/Kimi-K3-INT4-Slice8-TP8 \
  --num-layers 24
```
