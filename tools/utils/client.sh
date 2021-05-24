# usage: backupJobViaSSH "hackme" "15719703"
function backupJobViaSSH () {
    password=$1
    job_id=$2

    url="stfo194b@taurus.hrsk.tu-dresden.de"
    folder="/home/stfo194b/tesi/learnable-triangulation-pytorch/tools"

    full_path="${url}:${folder}/${job_id}.out"

    echo "backing up ${full_path} ..."
    sshpass -p ${password} scp ${full_path} .
}

# usage: monitorJobViaSSH "hackme" "15719703"
function monitorJobViaSSH () {
    password=$1
    job_id=$2

    sleep_seconds=120  # 2 minutes
    n_times=30  # 1 hour

    for n_time in $(seq 1 ${n_times}) ; do {
        now=$(date +%H:%M:%S)
        
        echo "${now}: updating local job file for ${job_id} ..."
        backupJobViaSSH ${password} ${job_id}
        
        echo "          done ${n_time}/${n_times} -> waiting ${sleep_seconds}\" ..."
        sleep ${sleep_seconds}
    }
    done
}