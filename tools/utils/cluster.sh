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
function slurmShowUsageInMonth () {
    account=$1
    resource=${2}
    max_hours=${3}

    day=$(date +"%d")
    month=$(date +"%m")
    year=$(date +"%Y")
    firstOfThisMonth=${year}-${month}-01
    daysInMonth=30

    bc_precision="scale=3"

    raw=$(sreport cluster AccountUtilizationByUser Accounts=${account} Start=${firstOfThisMonth} --tres=${resource})
    cpu_minutes=$(echo ${raw} | tail -n1 | awk '{print $6}')
    cpu_hours=$(echo "${bc_precision};${cpu_minutes}/60.0" | bc)
    as_perc=$(echo "${bc_precision};${cpu_hours}/${max_hours}*100.0" | bc)
    predicted_by_end=$(echo "${bc_precision};${daysInMonth}/${day}*${cpu_hours}" | bc)
    predicted_as_perc=$(echo "${bc_precision};${predicted_by_end}/${max_hours}*100.0" | bc)

    echo "${cpu_hours} ${resource}-hours used (since ${firstOfThisMonth}, ${as_perc} % of max)"
    echo "${predicted_by_end} ${resource}-hours will be used (by EOM, at this rate, ${predicted_as_perc} % of max)"
}
alias showClusterUsageInMonth='slurmShowUsageInMonth p_humanpose cpu 3500 && echo && slurmShowUsageInMonth p_humanpose gres/gpu 250'

# usage: showJobFolder "15901850"
function showJobFolder () {
    job_id=$1
    root_folder="/scratch/ws/0/stfo194b-p_humanpose/learnable-triangulation-pytorch/logs"

    job_folder=$(cat ${job_id}.out| grep Experiment | awk '{print $3}')

    echo ${root_folder}/${job_folder}
}

# usage: showJobConfig "15901850"
function showJobConfig () {
    job_id=$1
    job_folder=$(showJobFolder ${job_id})
    config_file=${job_folder}/config.yaml
    
    cat ${config_file}
}

#usage monatorCam2cam "16583749"
function monatorCam2cam () {
    job_id=$1

    watch -n5 "tail -n50 ${job_id}.out | grep -E 'cam2cam|epoc|loss' | grep -E 'comple|mm|loss'"
}