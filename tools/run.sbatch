#!/bin/bash

#SBATCH --partition=ml
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=04:00:00
#SBATCH --account=p_humanpose
#SBATCH --output=%j.out
#SBATCH --error=%j.err

date +"%Y-%m-%d %H:%M:%S"

source ./requirements.sh
source ./utils.sh

source /sw/installed/Anaconda3/2019.03/etc/profile.d/conda.sh
conda deactivate
conda activate ${KERNELS_DIR}/${KERNEL_NAME}  # or source
which python  # just to check

cd /home/stfo194b/tesi/learnable-triangulation-pytorch
pwd  # just to check

LOGS_FOLDER="/scratch/ws/0/stfo194b-p_humanpose/learnable-triangulation-pytorch/logs"
# LOGS_FOLDER="/lustre/ssd/ws/stfo194b-p_humanpose/learnable-triangulation-pytorch/logs"

EXP_CONFIG="experiments/human36m/train/human36m_alg.yaml"

python3 train.py --config ${EXP_CONFIG} --logdir ${LOGS_FOLDER}

# see latest results ...
# last_log=${LOGS_FOLDER}/$(ls ${LOGS_FOLDER} | tail -n1)
# checkpoints_folder=${last_log}
# seeMetrics ${checkpoints_folder}

# ... interactively
# watch -n5 "tree ${LOGS_FOLDER}/$(ls ${LOGS_FOLDER} | tail -n1)/checkpoints"

# eval: python3 train.py --eval --eval_dataset val --config experiments/human36m/eval/human36m_alg.yaml --logdir ${LOGS_FOLDER}

date +"%Y-%m-%d %H:%M:%S"