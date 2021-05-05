import re
import numpy as np

from mvn.utils.misc import drop_na


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
    """ parses seeMetrics /scratch/ws/... > wow.txt """

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
        print('= found {:d} / {:d} training / validation epochs'.format(
            len(train_metrics), len(eval_metrics)
        ))

    return train_metrics, eval_metrics


def parse_job_log(f_path, verbose=True):
    """ parses 14034239.out """

    lines = get_lines(f_path)

    exp_name = next(filter(
        lambda x: x.startswith('Experiment name:'),
        lines
    ))  # should be in the first lines -> just take 1st occurrence
    exp_name = exp_name.split()[-1]  # remove 'Experiment name:'

    train_data_amount = None
    eval_data_amount = None

    epochs = []  # will be a  [] of {} with details about each epoch
    current_epoch_details = {
        'epoch': None,
        'training loss / batch': [],
        'training metrics': None,
        'eval metrics': None
    }  # tmp epoch details

    for line in lines:
        if 'training dataset length:' in line:
            train_data_amount = float(line.split()[-1])

        if 'validation dataset length:' in line:
            eval_data_amount = float(line.split()[-1])

        if 'has started!' in line:  # new epoch
            try:
                current_epoch_details['epoch'] = int(line.split()[1])
            except:
                current_epoch_details['epoch'] = int(line.split()[2])
            
            current_epoch_details['training loss / batch'] = []

        if 'training batch iter' in line:  # batch loss
            loss = parse_fp_number(line.split('~')[-1])
            current_epoch_details['training loss / batch'].append(loss)

        if 'training MPJPE' in line:
            metric = parse_fp_number(line.split(':')[-1].split()[0])
            current_epoch_details['training metrics'] = metric

        if 'eval MPJPE' in line:
            metric = parse_fp_number(line.split(':')[-1].split()[0])
            current_epoch_details['eval metrics'] = metric

        if 'complete!' in line:  # end of epoch
            epochs.append(current_epoch_details.copy())

    if verbose:
        print('{} (from {})'.format(exp_name, f_path))
        print('training on {:.0f}, evaluating on {:.0f}'.format(
            train_data_amount,
            eval_data_amount
        ))
        print('found {:.0f} epochs'.format(len(epochs)))

        loss = [np.sum(epoch['training loss / batch']) for epoch in epochs]
        print('training loss in [{:.1f}, {:.1f}]'.format(
            np.min(drop_na(loss)),
            np.max(drop_na(loss))
        ))

        training_metrics = [epoch['training metrics'] for epoch in epochs]
        print('training metrics in [{:.1f}, {:.1f}]'.format(
            np.min(drop_na(training_metrics)),
            np.max(drop_na(training_metrics))
        ))

        eval_metrics = [epoch['eval metrics'] for epoch in epochs]
        print('eval metrics in [{:.1f}, {:.1f}]'.format(
            np.min(drop_na(eval_metrics)),
            np.max(drop_na(eval_metrics))
        ))

    return exp_name, train_data_amount, eval_data_amount, epochs
