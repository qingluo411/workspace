cd /root/workspace/LLVM
./build.sh
cd /root/workspace/DLCsim
./build.sh
cd /root/workspace/DLCSynapse
./compile.sh
cd /root/workspace/DLC_Custom_Kernel
./build.sh
cd /root/workspace/DLC_CL
./build.sh
cd /root/workspace/pytorch
rm -rf build/* && rm -rf dist/* && USE_CUDA=0 python3 setup.py clean
export PYTORCH_BUILD_VERSION=2.5.0 
export PYTORCH_BUILD_NUMBER=0 
USE_CUDA=0 DEBUG=0 python3 setup.py develop 