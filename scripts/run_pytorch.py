import argparse
import yaml
import os
import pprint
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


from pickle import dump
import warnings
warnings.filterwarnings('ignore')

import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger

from airquality import models as models
from airquality import datamodules as datamodules
from airquality.analysis.postprocessing import aggregate_effects, aggregate_n_params, aggregate_predictions, accuracy_report
from airquality.utils.dynamic_argument_parser import update_config_command_line
from airquality.models.pytorch import FCNN, model_summary

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

random_seed = 1
np.random.seed(random_seed)
random.seed(random_seed)

def create_run_config(args):
    base_config = {k: v for k, v, in vars(args).items() if not k == 'config_updates'}
    base_config['datamodule'] = datamodules.spec(base_config['datamodule_id']).default_config.copy()
    config = update_config_command_line(base_config, args.config_updates)
    return config

def run_cv_trial(datamodule, trial_dir):
    for ii in range(datamodule.k_fold):
        fold_dir = f'{trial_dir}/fold_{ii:02d}'
        os.makedirs(fold_dir, exist_ok=True)
        logger.info(f"Running CV fold {ii+1} / {datamodule.k_fold} and saving to {fold_dir}")

        # Initialize model
        model = FCNN(input_size=datamodule.shape[0][1])

        # Initialize trainer
        trainer = L.Trainer(
            max_epochs=100,
            gradient_clip_val=0.5,
            logger=CSVLogger(
                save_dir=config['save_dir'],
                name=os.path.basename(trial_dir),
                version=os.path.basename(fold_dir),
            )
        )

        predictions, summary_df = run_cv_fold(datamodule, model, trainer, k_fold_index=ii)
    
        # Save predictions, IQR scaling, and summary
        datamodule.iqr.to_csv(f'{fold_dir}/interquartile_range.csv')
        predictions.to_csv(f'{fold_dir}/model_predictions.csv', index=False)
        summary_df.to_csv(f'{fold_dir}/model_summary.csv')

        accuracy_report(predictions).to_csv(f'{fold_dir}/accuracy_report.csv')

def run_cv_fold(datamodule, model, trainer, k_fold_index=0):
    datamodule.k_fold_index = k_fold_index

    # Fit model on dataset
    trainer.fit(model, datamodule)

    # Evaluate predictions on test set
    predictions = datamodule.test_dataframe \
        .set_index(['date_local', 'ZCTA5CE20']) \
        .sort_index() \
        .rename({'visits': 'Observed'}, axis=1)

    model_predictions = trainer.predict(model, datamodule)
    predictions['Predicted'] = torch.cat(model_predictions, dim=0).cpu().numpy()
    
    # Perform statistical inference on trained model
    summary_df = model_summary(model, datamodule)

    # Remove IQR scaling for plotting purposes
    for key in datamodule.iqr.drop('visits').index:
        predictions[key] *= datamodule.iqr.loc[key]
    predictions['Observed'] *= datamodule.iqr['visits']
    predictions['Predicted'] *= datamodule.iqr['visits']
    predictions = predictions.reset_index()

    return predictions, summary_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datamodule_id', choices=datamodules.list_registered_modules(), default='pytorch_daily')
    parser.add_argument('--n_trials', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default=os.path.join(os.environ['MODELS_DIR'], 'test'),
                        help="Save location for model results"),
    parser.add_argument('--config_updates', nargs='*',
                        help="Update configuration with format [key]=[value]")
    args = parser.parse_args()

    config = create_run_config(args)
    os.makedirs(config["save_dir"], exist_ok=True)
    with open(f'{config["save_dir"]}/config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Load data
    datamodule = datamodules.make(args.datamodule_id, **config['datamodule'])
    datamodule.setup('none') # Just load the data but don't do train/test splitting

    # Load model
    if config['n_trials'] == 1:
        run_cv_trial(datamodule, config['save_dir'])
    
    else:
        for ii in range(config['n_trials']):
            trial_dir = f'{config["save_dir"]}/trial_{ii:02d}'
            os.makedirs(trial_dir, exist_ok=True)
            logger.info(f"Running CV trial {ii+1} / {config['n_trials']} with seed {datamodule.random_seed}")
            run_cv_trial(datamodule, trial_dir)

            # Increment random seed for the next trial
            datamodule.random_seed = datamodule.random_seed + 1
    
    # Aggregate across folds and trials to make plotting easier
    aggregate_predictions(config["save_dir"], overwrite=True)
    aggregate_effects(config["save_dir"], overwrite=True)