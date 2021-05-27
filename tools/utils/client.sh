# usage: backupJobViaSSH "hackme" "15719703"
function backupJobViaSSH () {
    password=$1
    job_id=$2

    url="stfo194b@taurus.hrsk.tu-dresden.de"
    folder="/home/stfo194b/tesi/learnable-triangulation-pytorch/tools"

    full_path="${url}:${folder}/${job_id}.out"

    echo "backing up ${full_path} ..."
    sshpass -p ${password} rsync --progress -avz ${full_path} .
}

# usage: backupExperimentViaSSH "hackme" "25.05.2021-18:51:38"
function backupExperimentViaSSH () {
    password=$1
    experiment_name=$2

    url="stfo194b@taurus.hrsk.tu-dresden.de"
    folder="/scratch/ws/0/stfo194b-p_humanpose/learnable-triangulation-pytorch/logs/human36m_alg_AlgebraicTriangulationNet@${experiment_name}/epoch-0-iter-0"

    full_path="${url}:${folder}"
    local_path="./human36m_alg_AlgebraicTriangulationNet@${experiment_name}/"
    mkdir -p ${local_path}

    echo "backing up ${full_path} -> ${local_path}"
    sshpass -p ${password} scp -r ${full_path} ${local_path}
}

# usage: monatorJobViaSSH "hackme" "15719703"
function monatorJobViaSSH () {
    password=$1
    job_id=$2

    sleep_seconds=60
    n_times=60

    for n_time in $(seq 1 ${n_times}) ; do {
        now=$(date +%H:%M:%S)
        
        echo "${now}: updating local job file for ${job_id} ..."
        backupJobViaSSH ${password} ${job_id}
        
        echo "          done ${n_time}/${n_times} -> waiting ${sleep_seconds}\" ..."
        sleep ${sleep_seconds}
    }
    done
}