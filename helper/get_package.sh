export LD_LIBRARY_PATH=/usr/local/chipltech/dlccl/lib:/usr/local/chipltech/simulator/lib:/usr/local/chipltech/synapse/lib:/usr/local/chipltech/kernel:/usr/local/chipltech/bluejay/build:${LD_LIBRARY_PATH:-}
export CUSTOM_KERNEL_PATH=/usr/local/chipltech/synapse/lib
export WHEELHOUSE=/root/workspace/wheelhouse
mkdir -p $WHEELHOUSE

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
cp /usr/local/chipltech/synapse/lib/libcustom_dlc_perf_lib.so $WHEELHOUSE
cd /root/workspace/DLC_CL
rm -rf build/
./build.sh
cd /root/workspace/pytorch
rm -rf build/* && rm -rf dist/* && USE_CUDA=0 python3 setup.py clean
export PYTORCH_BUILD_VERSION=2.5.0 
export PYTORCH_BUILD_NUMBER=0 
USE_CUDA=0 DEBUG=0 python3 setup.py develop 
USE_CUDA=0 DEBUG=0 python3 setup.py bdist_wheel --dist-dir $WHEELHOUSE
# vllm
cd /root/workspace/vllm
rm -rf build/* && rm -rf dist/* && python setup.py clean
VLLM_TARGET_DEVICE=empty \
VLLM_VERSION_OVERRIDE='0.30.1rc0+dlc' \
VLLM_SKIP_VERSION_SUFFIX=1 \
MAX_JOBS=$(nproc) \
python setup.py bdist_wheel \
  --dist-dir $WHEELHOUSE

# vLLM-CL：编译 DLC C++ 扩展
cd /root/workspace/vllm-cl
rm -rf build/* && rm -rf dist/* && python setup.py clean
SETUPTOOLS_SCM_PRETEND_VERSION='0.30.1rc0+dlc' \
MAX_JOBS=$(nproc) \
CMAKE_BUILD_TYPE=Release \
COMPILE_CUSTOM_KERNELS=1 \
python setup.py bdist_wheel \
  --dist-dir $WHEELHOUSE


cd /root/workspace/torchaudio-2.5

CXXFLAGS="-I$DLC_INC" \
CPLUS_INCLUDE_PATH="$DLC_INC" \
BUILD_VERSION="2.5.0+dlc" \
PYTORCH_VERSION="2.5.0" \
USE_CUDA=0 \
BUILD_CUDA_CTC_DECODER=0 \
USE_FFMPEG=0 \
BUILD_SOX=0 \
MAX_JOBS=8 \
CMAKE_BUILD_PARALLEL_LEVEL=8 \
"$PY" setup.py bdist_wheel \
  --dist-dir "$WHEELHOUSE"


BUILDDEPS=$(mktemp -d /tmp/torchvision-builddeps.XXXXXX)
"$PY" -m pip install --target "$BUILDDEPS" 'setuptools<81' wheel

cd /root/workspace/torchvision

PYTHONPATH="$BUILDDEPS${PYTHONPATH:+:$PYTHONPATH}" \
CXXFLAGS="-I$DLC_INC" \
CPLUS_INCLUDE_PATH="$DLC_INC" \
BUILD_VERSION="0.19.0+dlc" \
PYTORCH_VERSION="2.5.0" \
FORCE_CUDA=0 \
FORCE_MPS=0 \
TORCHVISION_USE_PNG=0 \
TORCHVISION_USE_JPEG=0 \
TORCHVISION_USE_NVJPEG=0 \
TORCHVISION_USE_FFMPEG=0 \
TORCHVISION_USE_VIDEO_CODEC=0 \
MAX_JOBS=8 \
"$PY" setup.py bdist_wheel \
  --dist-dir "$WHEELHOUSE"