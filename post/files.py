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


def parse_job_log(f_path, verbose=False):
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
        'total loss / batch': [],
        'training metrics': None,
        'eval metrics': None
    }  # tmp epoch details
    lr_reductions = []

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
            
            for key in current_epoch_details:
                if 'loss' in key:
                    current_epoch_details[key] = []

        if 'training batch iter' in line and (not 'MPJPE' in line):  # batch loss
            tokens = line.split('~')

            # format used in https://github.com/sirfoga/learnable-triangulation-pytorch/blob/master/train.py#L627

            if len(tokens) > 2:
                loss = parse_fp_number(tokens[1].split(',')[0])
                key = 'geodesic loss / batch'
                if key not in current_epoch_details:
                    current_epoch_details[key] = []
                current_epoch_details[key].append(loss)

            if len(tokens) > 3:
                loss = parse_fp_number(tokens[2].split(',')[0])
                key = 'L2 on T loss / batch'
                if key not in current_epoch_details:
                    current_epoch_details[key] = []
                current_epoch_details[key].append(loss)

            if len(tokens) > 4:
                loss = parse_fp_number(tokens[3].split(',')[0])
                key = 'L2 proj loss / batch'
                if key not in current_epoch_details:
                    current_epoch_details[key] = []
                current_epoch_details[key].append(loss)
                
            if len(tokens) > 5:
                loss = parse_fp_number(tokens[4].split(',')[0])
                key = 'L2 on R loss / batch'
                if key not in current_epoch_details:
                    current_epoch_details[key] = []
                current_epoch_details[key].append(loss)
                
            if len(tokens) > 6:
                loss = parse_fp_number(tokens[5].split(',')[0])
                key = 'L2 on 3D loss / batch'
                if key not in current_epoch_details:
                    current_epoch_details[key] = []
                current_epoch_details[key].append(loss)
                
            if len(tokens) > 7:
                loss = parse_fp_number(tokens[6].split(',')[0])
                key = 'self-consistency R loss / batch'
                if key not in current_epoch_details:
                    current_epoch_details[key] = []
                current_epoch_details[key].append(loss)
                
            if len(tokens) > 8:
                loss = parse_fp_number(tokens[7].split(',')[0])
                key = 'self-consistency t loss / batch'
                if key not in current_epoch_details:
                    current_epoch_details[key] = []
                current_epoch_details[key].append(loss)
                
            if len(tokens) > 9:
                loss = parse_fp_number(tokens[8].split(',')[0])
                key = 'self-consistency proj loss / batch'
                if key not in current_epoch_details:
                    current_epoch_details[key] = []
                current_epoch_details[key].append(loss)
                
            if len(tokens) > 10:
                loss = parse_fp_number(tokens[9].split(',')[0])
                key = 'self-consistency DLT loss / batch'
                if key not in current_epoch_details:
                    current_epoch_details[key] = []
                current_epoch_details[key].append(loss)
                
            if len(tokens) > 11:
                loss = parse_fp_number(tokens[10].split(',')[0])
                key = 'self-consistency pivot loss / batch'
                if key not in current_epoch_details:
                    current_epoch_details[key] = []
                current_epoch_details[key].append(loss)

            total_loss = parse_fp_number(tokens[-1])
            current_epoch_details['total loss / batch'].append(total_loss)

        if 'training MPJPE' in line:
            metric = parse_fp_number(line.split(':')[-1].split()[0])
            current_epoch_details['training metrics'] = metric

        if 'eval MPJPE' in line:
            metric = parse_fp_number(line.split(':')[-1].split()[0])
            current_epoch_details['eval metrics'] = metric

        if 'complete!' in line:  # end of epoch
            epochs.append(current_epoch_details.copy())

        if 'reducing learning rate' in line:
            tokens = line[:-1].split()

            lr_reductions.append({
                'epoch': current_epoch_details['epoch'],
                'lr': parse_fp_number(tokens[-1])
            })

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

    return exp_name, train_data_amount, eval_data_amount, epochs, lr_reductions
