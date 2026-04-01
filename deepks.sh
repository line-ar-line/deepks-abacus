conda create -n deepks python=3.9 numpy scipy h5py ruamel.yaml paramiko -c conda-forge 
conda activate deepks
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# numpy的版本必须是1.xx，否则会有问题   
conda install numpy==1.24 -c conda-forge
pip install ruamel.yaml==0.17.21
# the conda package does not support python >= 3.8 so we use pip
pip install pyscf
pip install psutil
#pip install git+https://github.com/deepmodeling/deepks-kit/develop.git
# 1. 克隆仓库
#git clone -b develop https://github.com/deepmodeling/deepks-kit.git
#cd deepks-kit
#pip install . 
# 或者
git clone -b develop https://github.com/MCresearch/DeePKS-L.git
cd DeePKS-L
pip install .