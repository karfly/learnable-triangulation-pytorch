KERNEL_NAME="learnable-triangulation-pytorch-alpha"
KERNELS_DIR=/home/${USER}/kernels/

module --force purge
module load modenv/ml
module load PythonAnaconda/3.7

# optional: load the following to test if packages conflict
#should be installed in env module load PyTorch
module load scikit-learn
module load matplotlib
module load Pillow
module load h5py/2.10
