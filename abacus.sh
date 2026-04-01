#!/bin/bash
set -e  # 有错误立即退出
mypath=/home/linearline/project

# # 安装依赖库
# sudo apt update 
# sudo apt install -y libopenblas-openmp-dev
# sudo apt install -y liblapack-dev 
# sudo apt install -y libfftw3-dev
# sudo apt install -y libopenmpi-dev
# sudo apt install -y libscalapack-mpi-dev
# sudo apt install -y libcereal-dev
# sudo apt install -y libelpa-dev
# #  从github获取ABACUS
# cd "${mypath}"
# git clone https://github.com/deepmodeling/abacus-develop.git


#如果用oneapi
#source /opt/intel/oneapi/2024.1/oneapi-vars.sh

# # # 设置所有临时目录到数据盘
# cd "${mypath}"
# mkdir tmp
# mkdir pip_cache
# export PIP_CACHE_DIR=$(pwd)/pip_cache
# export TMPDIR=$(pwd)/tmp
# export TEMP=$(pwd)/tmp
# export TMP=$(pwd)/tmp



# # libxc 下载地址https://libxc.gitlab.io/download/， /home/linearline/project/libxc-7.0.0.tar.bz2
# echo "编译Libxc"
# wget https://gitlab.com/libxc/libxc/-/archive/7.0.0/libxc-7.0.0.tar.bz2
# tar -xjvf libxc-7.0.0.tar.bz2
# cd libxc-7.0.0
# cmake -B build -DCMAKE_INSTALL_PREFIX=${mypath}/install
# # 我的cmake版本太新4.3.1，会报错。此时应该改为cmake -B build -DCMAKE_POLICY_VERSION_MINIMUM=3.5 
# # 或者手动改libxc的cmakelist的文件
# cd build && make 
# #make test
# make install


# # 下载LibRI,LibComm模块
# echo "下载LibRI,LibComm模块"

# cd "${mypath}"
# #git clone https://github.com/abacusmodeling/LibRI.git
# git clone https://github.com/abacusmodeling/LibComm.git

# # 编译ABACUS杂化版本
# echo "编译ABACUS杂化版本"

cd "${mypath}/abacus-develop"
# #CXX=icpx 
cmake -B build -DGIT_SUBMODULE=OFF -DLIBRI_DIR=${mypath}/LibRI -DLIBCOMM_DIR=${mypath}/LibComm -DLibxc_DIR=${mypath}/libxc
cd build && make 
#-j`nproc`

# cd "${mypath}"
# echo "克隆libnpy到本地"
# git clone https://github.com/llohse/libnpy.git


# echo "安装libtorch"
## CPU 版本
# wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.11.0%2Bcpu.zip
# unzip project/libtorch-shared-with-deps-2.11.0+cpu.zip
# 
## GPU
# pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118 --target "${mypath}/libtorch"

# echo "编译MLALGO版本"
# cd "${mypath}/abacus-develop"
# cmake -B build -DENABLE_MLALGO=1 -DTorch_DIR=${mypath}/libtorch/share/cmake/Torch/ -Dlibnpy_INCLUDE_DIR=${mypath}/libnpy/include
# cd build && make
