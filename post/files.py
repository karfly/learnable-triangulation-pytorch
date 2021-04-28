import re


def get_lines(f_path):
    lines = []

    with open(f_path, 'r') as reader:
        lines = reader.readlines()
        lines = [
            line.strip()
            for line in lines
        ]

    return lines


def is_fp_number(x):
    return not (re.match('\d.', x) is None)


def parse_fp_number(x):
    return float(x)


def parse_metrics_log(f_path, verbose=True):

    lines = get_lines(f_path)
    split_index = lines.index('... on EVALuation set')
    training_stop_index = split_index - 1
    eval_start_index = split_index + 1

    train_metrics = map(
        parse_fp_number,
        filter(is_fp_number, lines[:training_stop_index])
    )

    eval_metrics = map(
        parse_fp_number,
        filter(is_fp_number, lines[eval_start_index:])
    )

    train_metrics, eval_metrics = list(train_metrics), list(eval_metrics)

    if verbose:
        print('found {:d} / {:d} training / validation epochs'.format(
            len(train_metrics), len(eval_metrics)
        ))

    return train_metrics, eval_metrics


def parse_job_log(f_path):
    lines = get_lines(f_path)

    exp_name = next(filter(
        lambda x: x.startswith('Experiment name:'),
        lines[:50]
    ))  # should be in the first lines -> just take 1st occurrence
    exp_name = exp_name.split()[-1]  # remove 'Experiment name:'

    epochs = []  # will be a  [] of {} with details about each epoch
    current_epoch_details = {
        'epoch': None,
        'training loss / batch': [],
        'training metrics': None,
        'eval metrics': None
    }  # tmp epoch details

    for line in lines:
        if line.endswith('has started!'):  # new epoch
            current_epoch_details['epoch'] = int(line.split()[1])
            current_epoch_details['training loss / batch'] = []
        
        if line.startswith('training batch iter'):  # batch loss
            loss = parse_fp_number(line.split('~')[-1])
            current_epoch_details['training loss / batch'].append(loss)

        if line.startswith('training MPJPE'):
            metric = parse_fp_number(line.split(':')[-1].split()[0])
            current_epoch_details['training metrics'] = metric

        if line.startswith('eval MPJPE'):
            metric = parse_fp_number(line.split(':')[-1].split()[0])
            current_epoch_details['eval metrics'] = metric

        if line.endswith('complete!'):  # end of epoch
            epochs.append(current_epoch_details.copy())

    return exp_name, epochs
