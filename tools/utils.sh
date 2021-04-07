function getAverage() {
    f=$1
    head -n4 ${f} | tail -n1 | awk '{print $2}' | sed 's/,//';
}

function seeMetrics () {
    checkpoints_folder=$1

    echo "metrics on TRAINing sets"
    for f in $(find ${checkpoints_folder} -name "metric_train.json" | sort);
    do
        getAverage ${f}
    done
    echo

    echo "metrics on EVALuation sets"
    for f in $(find ${checkpoints_folder} -name "metric_eval.json" | sort);
    do
        getAverage ${f}
    done
    echo
}
