export LD_LIBRARY_PATH=/usr/local/chipltech/dlccl/lib:/usr/local/chipltech/simulator/lib:/usr/local/chipltech/synapse/lib:/usr/local/chipltech/kernel:/usr/local/chipltech/bluejay/build:${LD_LIBRARY_PATH:-}
export CUSTOM_KERNEL_PATH=/usr/local/chipltech/synapse/lib
cd /root/workspace/LLVM
rm -rf build/
./build.sh
cd /root/workspace/DLCsim
rm -rf build/
./build.sh
cd /root/workspace/DLCSynapse
rm -rf build/
./compile.sh
cd /root/workspace/DLC_Custom_Kernel
rm -rf build/
./build.sh
cd /root/workspace/DLC_CL
rm -rf build/
./build.sh
cd /root/workspace/pytorch
rm -rf build/* && rm -rf dist/* && USE_CUDA=0 python3 setup.py clean
export PYTORCH_BUILD_VERSION=2.5.0 
export PYTORCH_BUILD_NUMBER=0 
USE_CUDA=0 DEBUG=0 python3 setup.py develop 

# vllm
cd /root/workspace/
VLLM_TARGET_DEVICE=empty uv pip install -v -e vllm --no-build-isolation
uv pip install -v -e vllm-cl --no-build-isolation
