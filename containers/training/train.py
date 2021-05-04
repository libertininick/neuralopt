import argparse
import glob
import json
import os
import re
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils import PriceSeriesDataset
from modules import PriceSeriesFeaturizer
from training_utils import feature_learning_loss, train_feat_batch, lr_schedule, set_lr

# Argument defaults
arg_definitions = {
    'input_dir': {
        'flag': '--input_dir', 
        'type': str, 
        'default': '/opt/ml/input/data/all'
    },
    'model_dir': {
        'flag': '--model_dir', 
        'type': str, 
        'default': '/opt/ml/model'
    },
    'output_dir': {
        'flag': '--output_dir', 
        'type': str, 
        'default': '/opt/ml/output'
    },
    'p_valid': {
        'flag': '--p_valid', 
        'type': float, 
        'default': 0.050
    },
    'p_test': {
        'flag': '--p_test', 
        'type': float, 
        'default': 0.025
    },
    'historical_seq_len': {
        'flag': '--historical_seq_len', 
        'type': int, 
        'default': 512
    },
    'future_seq_len': {
        'flag': '--future_seq_len', 
        'type': int, 
        'default': 32
    },
    'n_dist_targets': {
        'flag': '--n_dist_targets', 
        'type': int, 
        'default': 51
    },
    'n_features': {
        'flag': '--n_features', 
        'type': int, 
        'default': 64
    },
    'batch_size': {
        'flag': '--batch_size', 
        'type': int, 
        'default': 16
    },
    'epochs': {
        'flag': '--epochs', 
        'type': int, 
        'default': 1
    },
}


def define_hyperparameters():
    parser = argparse.ArgumentParser()
    for v in arg_definitions.values():
        parser.add_argument(v['flag'], type=v['type'], default=v['default'])
    return parser


def create_datasets(
    data_dir, 
    p_valid, 
    p_test,
    historical_seq_len,
    future_seq_len,
    n_dist_targets,
    seed=1234):
    """Splits files into training, validation and test
    """
    p = re.compile(r'[^\\\\|\/]{1,100}(?=\.pkl$)')

    files = np.array(glob.glob(f'{data_dir}/*.pkl'))
    symbols = np.array([p.findall(file)[0] for file in files])

    rnd = np.random.RandomState(seed)

    n = len(files)
    n_valid = int(n*p_valid)
    n_test = int(n*p_test)
    n_train = n - n_valid - n_test

    idxs_all = np.arange(n)
    idxs_train = rnd.choice(idxs_all, n_train, replace=False)
    idxs_other = np.setdiff1d(idxs_all, idxs_train)
    idxs_valid = rnd.choice(idxs_other, n_valid, replace=False)
    idxs_test = np.setdiff1d(idxs_other, idxs_valid)

    datasets = {
        key: PriceSeriesDataset(
            symbols=symbols[idxs],
            files=files[idxs],
            n_historical=historical_seq_len,
            n_future=future_seq_len,
            n_dist_targets=n_dist_targets,
        )
        for key, idxs
        in [
            ('train', idxs_train), 
            ('valid', idxs_valid), 
            ('test', idxs_test)
        ]
    }

    return datasets


def main():
    # Hyperparameters
    hyperparm_path = '/opt/ml/input/config/hyperparameters.json'
    if os.path.exists(hyperparm_path):
        # Parse parmaters from hyperparameters.json
        with open(hyperparm_path, 'r') as fp:
            args = json.load(fp)
    else:
        # Parse parameters from command line
        args, _ = define_hyperparameters().parse_known_args()
        args = vars(args)

    # Argument getter
    def get_arg(key):
        definition = arg_definitions.get(key)

        if definition is None:
            return None
        else:
            try:
                value = args.get(key, definition['default'])
                return definition['type'](value)
            except:
                return definition['default']

    # Log
    out_dir = get_arg('output_dir')
    st = time()
    log = f'{(time() - st)/60:>7.2f}m: Hyperparameters parsed\n'

    # Create datasets
    try:
        datasets = create_datasets(
            data_dir=get_arg('input_dir'), 
            p_valid=get_arg('p_valid'), 
            p_test=get_arg('p_test'),
            historical_seq_len=get_arg('historical_seq_len'),
            future_seq_len=get_arg('future_seq_len'),
            n_dist_targets=get_arg('n_dist_targets'),
        )
        log += f'{(time() - st)/60:>7.2f}m: Datasets created\n'
    except Exception as e:
        log += f'{(time() - st)/60:>7.2f}m: Dataset FAILED\n'
        log += f'Exception: {str(e)}'
        with open(f'{out_dir}/log.txt','a') as fp:
            fp.write(log)
        raise

    # Set device
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    log += f'{(time() - st)/60:>7.2f}m: Device set to {device}\n'

    # Define model
    try:
        model = PriceSeriesFeaturizer(
            n_features=get_arg('n_features'),
            historical_seq_len=get_arg('historical_seq_len'),
            future_seq_len=get_arg('future_seq_len'),
            n_dist_targets=get_arg('n_dist_targets'),
        )
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters())
        log += f'{(time() - st)/60:>7.2f}m: Model defined\n'
    except Exception as e:
        log += f'{(time() - st)/60:>7.2f}m: Model FAILED\n'
        log += f'Exception: {str(e)}'
        with open(f'{out_dir}/log.txt','a') as fp:
            fp.write(log)
        raise

    with open(f'{out_dir}/log.txt','a') as fp:
        fp.write(log)

    # Training loop
    batch_size = get_arg('batch_size')
    cycle_len = 10000//batch_size
    lrs = lr_schedule(
        n_steps=len(datasets['train'])//batch_size + 1, 
        lr_min=0.00001, 
        lr_max=0.003*batch_size/128
    )
    loss_quantiles = [0.05,0.25,0.5,0.75,0.95]

    best_train_loss, best_valid_loss = np.inf, np.inf
    model_dir = get_arg('model_dir')
    
    for epoch in range(get_arg('epochs')):
        # Training pass
        model.train()
        train_losses = {'recon': [], 'LCH': [], 'dists': [], 'total': []}
        loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=3)
        for b_i, batch in enumerate(loader):
            # Train on batch
            try:
                set_lr(optimizer, lrs[b_i])
                losses_i = train_feat_batch(model, optimizer, feature_learning_loss, batch, device)
            except Exception as e:
                log = f'{(time() - st)/60:>7.2f}m: Batch {epoch:>3} {b_i + 1:<10,} FAILED\n'
                log += f'Exception: {str(e)}'
                with open(f'{out_dir}/log.txt','a') as fp:
                    fp.write(log)
                raise

            # Extract losses
            for ll, l_i in zip(train_losses.values(), losses_i):
                ll.append(l_i)

            if (b_i + 1) % cycle_len == 0:
                progress = (
                    f'{(time() - st)/60:>7.2f}m: {epoch:>3} {b_i + 1:<10,}' +
                    f''' { np.quantile(train_losses['recon'][-cycle_len:], q=loss_quantiles).round(3)}''' +
                    f''' { np.quantile(train_losses['LCH'][-cycle_len:], q=loss_quantiles).round(3)}''' + 
                    f''' { np.quantile(train_losses['dists'][-cycle_len:], q=loss_quantiles).round(3)}'''
                )
                with open(f'{model_dir}/training_progress.txt','a') as fp:
                    fp.write(f'{progress}\n')

        # Validation pass
        model.eval()
        valid_losses = {'recon': [], 'LCH': [], 'dists': [], 'total': []}
        loader = DataLoader(datasets['valid'], batch_size=batch_size, shuffle=True, num_workers=3)
        for b_i, batch in enumerate(loader):
            # Evaluate on batch
            try:
                with torch.no_grad():
                    *component_losses, total_loss = feature_learning_loss(model, batch, device)
            except Exception as e:
                log = f'{(time() - st)/60:>7.2f}m: Validation batch {epoch:>3} {b_i + 1:<10,} FAILED\n'
                log += f'Exception: {str(e)}'
                with open(f'{out_dir}/log.txt','a') as fp:
                    fp.write(log)
                raise
            
            # Extract losses
            losses_i = (*component_losses, total_loss.item())
            for ll, l_i in zip(valid_losses.values(), losses_i):
                ll.append(l_i)

        progress = (
            f'{(time() - st)/60:>7.2f}m: {epoch:>3} Validation' +
            f''' { np.quantile(valid_losses['recon'], q=loss_quantiles).round(3)}''' +
            f''' { np.quantile(valid_losses['LCH'], q=loss_quantiles).round(3)}''' + 
            f''' { np.quantile(valid_losses['dists'], q=loss_quantiles).round(3)}'''
        )
        
        # Checkpoint
        train_loss, valid_loss = np.mean(train_losses['total']), np.mean(valid_losses['total'])
        if train_loss <= best_train_loss and valid_loss <= best_valid_loss:
            # Save weights
            try:
                model.to('cpu')
                torch.save(model.state_dict(), f'{model_dir}/wts.pth')
                model.to(device)
            except Exception as e:
                log = f'{(time() - st)/60:>7.2f}m: Model save {epoch:>3} FAILED\n'
                log += f'Exception: {str(e)}'
                with open(f'{out_dir}/log.txt','a') as fp:
                    fp.write(log)
                raise
            
            # Update best losses
            best_train_loss, best_valid_loss = train_loss, valid_loss
            
            progress += '\n*** Model saved'
        else:
            progress += '\n!!! Model NOT saved'

        with open(f'{model_dir}/training_progress.txt','a') as fp:
            fp.write(f'{progress}\n')


if __name__ == "__main__":
    main()