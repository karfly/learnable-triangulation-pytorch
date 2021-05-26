srun \
--partition=ml \
--nodes=1 \
--tasks=1 \
--cpus-per-task=1 \
--gres=gpu:1 \
--mem-per-cpu=2048 \
--time=00:30:00 \
--account=p_humanpose \
--pty zsh

source ./requirements.sh

conda create --prefix ${KERNELS_DIR}/${KERNEL_NAME} python=3.7.4
conda deactivate
conda activate ${KERNELS_DIR}/${KERNEL_NAME}  # or source
which python  # just to check 

# optional: install kernel
# conda install ipykernel
# python -m ipykernel install --user --name "${KERNEL_NAME}"

# optional: install other packages (don't need to be in srun)
# source /sw/installed/Anaconda3/2019.03/etc/profile.d/conda.sh
# conda activate ${KERNELS_DIR}/${KERNEL_NAME}
# conda install -y ...

# optional: try importing packages
# python

conda deactivate
