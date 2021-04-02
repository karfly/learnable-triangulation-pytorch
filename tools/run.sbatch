#!/bin/bash

#SBATCH --partition=ml
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=02:00:00
#SBATCH --account=p_humanpose
#SBATCH --output=%j.out
#SBATCH --error=%j.err

source ./requirements.sh

source /sw/installed/Anaconda3/2019.03/etc/profile.d/conda.sh
conda deactivate
conda activate ${KERNELS_DIR}/${KERNEL_NAME}  # or source
which python  # just to check

# todo use /lustre/ssd/ws/stfo194b-p_humanpose

cd /home/stfo194b/tesi/learnable-triangulation-pytorch
pwd  # just to check

LOGS_FOLDER="/scratch/ws/0/stfo194b-p_humanpose/learnable-triangulation-pytorch/logs"

python3 train.py --config experiments/human36m/train/human36m_alg.yaml --logdir ${LOGS_FOLDER}

# see latest results
last_log=${LOGS_FOLDER}/$(ls ${LOGS_FOLDER} | tail -n1)
tree ${last_log}

checkpoints_folder=${last_log}
for f in $(find ${checkpoints_folder} -name "*.json" | sort);
    do head -n4 ${f} | tail -n1;
done

# eval: python3 train.py --eval --eval_dataset val --config experiments/human36m/eval/human36m_alg.yaml --logdir ${LOGS_FOLDER}
