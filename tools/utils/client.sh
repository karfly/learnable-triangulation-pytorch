# usage: backupJobViaSSH "hackme" "15719703"
function backupJobViaSSH () {
    password=$1
    job_id=$2

    url="stfo194b@taurus.hrsk.tu-dresden.de"
    folder="/home/stfo194b/tesi/learnable-triangulation-pytorch/tools"

    full_path="${url}:${folder}/${job_id}.out"

    echo "${full_path} ----> ${PWD}"
    sshpass -p ${password} scp ${full_path} .
}