function getAverage() {
    f=$1
    head -n4 ${f} | tail -n1 | awk '{print $2}' | sed 's/,//';
}

function seeMetrics () {
    checkpoints_folder=$1

    echo "... on TRAINing set\n"
    for f in $(find ${checkpoints_folder} -name "metric_train.json" | sort);
    do
        echo $f && getAverage ${f}
    done
    echo

    echo "... on EVALuation set\n"
    for f in $(find ${checkpoints_folder} -name "metric_eval.json" | sort);
    do
        getAverage ${f}
    done
    echo
}

function getLastJob () {
    squeue -u ${USER} | tail -n 1 | awk "{print \$1}"
}

function showLastOut () {
    cat $(getLastJob).out
}

function showLastErr () {
    cat $(getLastJob).err
}