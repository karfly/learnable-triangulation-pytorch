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

# usage: showLRReductions "16932350"
function showLRReductions () {
    job_id=$1
    log_file=${job_id}.out

    grep -i red ${log_file}
}

# from fs last job (`squeue` takes takes too much)
function getLastJob () {
    ls *.out | sort | tail -n1 | sed -e 's/.out//g'
}

function peekLastJob () {
    filename=$(getLastJob).out
    tail -n200 ${filename}

    echo "\n... from ${filename}"
}