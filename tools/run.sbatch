srun \
--partition=ml \
--nodes=1 \
--tasks=1 \
--cpus-per-task=1 \
--gres=gpu:1 \
--mem-per-cpu=2048 \
--time=00:30:00 \
--account=p_humanpose \
--pty bash 

source ./requirements.sh

source /sw/installed/Anaconda3/2019.03/etc/profile.d/conda.sh
conda deactivate
conda activate ${KERNELS_DIR}/${KERNEL_NAME}  # or source
which python  # just to check

