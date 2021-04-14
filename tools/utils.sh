function getErrorAbsolute() {
    f=$1
    head -n4 ${f} | tail -n1 | awk '{print $2}' | sed 's/,//';
}

function getErrorRelativeToPelvis() {
    f=$1
    head -n150 ${f} | tail -n1 | awk '{print $2}' | sed 's/,//';
}

function seeMetrics () {
    checkpoints_folder=$1

    echo "... on TRAINing set\n"
    for f in $(find ${checkpoints_folder} -name "metric_train.json" | sort);
    do
        # getErrorAbsolute ${f}
        getErrorRelativeToPelvis ${f}
    done
    echo

    echo "... on EVALuation set\n"
    for f in $(find ${checkpoints_folder} -name "metric_eval.json" | sort);
    do
        # getErrorAbsolute ${f}
        getErrorRelativeToPelvis ${f}
    done
    echo
}

# e.g backupLog '/scratch/ws/0/stfo194b-p_humanpose/learnable-triangulation-pytorch/logs/human36m_alg_AlgebraicTriangulationNet@12.04.2021-17:01:07' '/projects/p_humanpose/learnable-triangulation/good_logs'
function backupLog () {
    src_folder=$1
    backup_folder=$2
    dest_folder=${backup_folder}/$(basename ${src_folder})

    # copy all metrics = everything except checkpoints and results
    rsync -vrtlS --size-only --exclude '*.pth' --exclude '*.pkl' --exclude 'tb' ${src_folder} ${backup_folder}

    # copy everything from last checkpoint (useful for continuing training)
    last_ckp=$(find ${src_folder}/checkpoints -maxdepth 1 -type d | sort | tail -n1)
    cp -r ${last_ckp} ${dest_folder}/checkpoints

    du -sh ${dest_folder}
}