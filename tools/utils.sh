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

# usage: backupLog '/scratch/ws/0/stfo194b-p_humanpose/learnable-triangulation-pytorch/logs/human36m_alg_AlgebraicTriangulationNet@12.04.2021-17:01:07' '/projects/p_humanpose/learnable-triangulation/good_logs'
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

# usage: backupH36Data S11
function backupScratchData () {
    subject=$1
    src_folder=/scratch/ws/0/stfo194b-p_humanpose/h36m-fetch/processed/${subject}
    target_folder=/projects/p_humanpose/learnable-triangulation/data/human36m/processed/${subject}

    echo "now target ~ $(du -sh ${target_folder})"

    for action in $(ls ${src_folder});
    do
        src_action=${src_folder}/${action}
        cp -rv ${src_action} ${target_folder}
    done
    
    echo "now target ~ $(du -sh ${target_folder})"
    echo "src ~ $(du -sh ${src_folder})"
}

# usage: showClusterUsageInMonth "p_humanpose"
function showClusterUsageInMonth () {
    account=$1
    allocated_cpu_hours=3500

    day=$(date +"%d")
    month=$(date +"%m")
    year=$(date +"%Y")
    firstOfThisMonth=${year}-${month}-01
    daysInMonth=30

    bc_precision="scale=3"

    cpu_minutes=$(sreport cluster AccountUtilizationByUser Accounts=${account} Start=${firstOfThisMonth} | tail -n1 | awk '{print $5}')
    cpu_hours=$(echo "${bc_precision};${cpu_minutes}/60.0" | bc)
    as_perc=$(echo "${bc_precision};${cpu_hours}/${allocated_cpu_hours}*100.0" | bc)
    predicted_by_end=$(echo "${bc_precision};${daysInMonth}/${day}*${cpu_hours}" | bc)
    predicted_as_perc=$(echo "${bc_precision};${predicted_by_end}/${allocated_cpu_hours}*100.0" | bc)

    echo "${cpu_hours} CPU-hours used (since ${firstOfThisMonth}, ${as_perc} % of max)"
    echo "${predicted_by_end} CPU-hours will be used (by EOM, at this rate, ${predicted_as_perc} % of max)"
}