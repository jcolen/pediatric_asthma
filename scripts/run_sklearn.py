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

from airquality import models as models
from airquality import datamodules as datamodules
from airquality.analysis.postprocessing import aggregate_effects, aggregate_n_params, aggregate_predictions, accuracy_report
from airquality.utils.dynamic_argument_parser import update_config_command_line

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

random_seed = 1
np.random.seed(random_seed)
random.seed(random_seed)

def create_run_config(args):
    base_config = {k: v for k, v, in vars(args).items() if not k == 'config_updates'}
    base_config['model'] = models.spec(base_config['model_id']).default_config.copy()
    base_config['datamodule'] = datamodules.spec(base_config['datamodule_id']).default_config.copy()
    config = update_config_command_line(base_config, args.config_updates)
    return config

def run_cv_trial(datamodule, model, trial_dir):
    for ii in range(datamodule.k_fold):
        fold_dir = f'{trial_dir}/fold_{ii:02d}'
        os.makedirs(fold_dir, exist_ok=True)
        logger.info(f"Running CV fold {ii+1} / {datamodule.k_fold} and saving to {fold_dir}")

        predictions, summary_df = run_cv_fold(datamodule, model, k_fold_index=ii)

        # Save the model
        with open(f'{fold_dir}/model.pkl', 'wb') as f:
            dump(model, f, protocol=5)
    
        # Save predictions, IQR scaling, and summary
        datamodule.iqr.to_csv(f'{fold_dir}/interquartile_range.csv')
        predictions.to_csv(f'{fold_dir}/model_predictions.csv', index=False)
        summary_df.to_csv(f'{fold_dir}/model_summary.csv')

        accuracy_report(predictions).to_csv(f'{fold_dir}/accuracy_report.csv')

def run_cv_fold(datamodule, model, k_fold_index=0):
    datamodule.k_fold_index = k_fold_index
    datamodule.setup('fit')

    # Fit model on train dataset
    train_data = datamodule.get_train_data()
    model.fit(**train_data)
    
    # Perform statistical inference on trained model
    summary_df = model.summary(**train_data)

    # Evaluate predictions on test set
    test_data = datamodule.get_test_data()
    predictions = datamodule.test_dataframe \
        .set_index(['date_local', 'ZCTA5CE20']) \
        .sort_index() \
        .rename({'visits': 'Observed'}, axis=1)
    predictions['Predicted'] = model.predict(**test_data)

    # Remove IQR scaling for plotting purposes
    for key in datamodule.iqr.drop('visits').index:
        predictions[key] *= datamodule.iqr.loc[key]
    predictions['Observed'] *= datamodule.iqr['visits']
    predictions['Predicted'] *= datamodule.iqr['visits']
    predictions = predictions.reset_index()

    return predictions, summary_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', choices=models.list_registered_modules(), default='gam')
    parser.add_argument('--datamodule_id', choices=datamodules.list_registered_modules(), default='airquality_coi_daily')
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
    model = models.make(
        args.model_id,
        **config['model'],
        datamodule=datamodule,
    )

    if config['n_trials'] == 1:
        run_cv_trial(datamodule, model, config['save_dir'])
    
    else:
        for ii in range(config['n_trials']):
            trial_dir = f'{config["save_dir"]}/trial_{ii:02d}'
            os.makedirs(trial_dir, exist_ok=True)
            logger.info(f"Running CV trial {ii+1} / {config['n_trials']} with seed {datamodule.random_seed}")
            run_cv_trial(datamodule, model, trial_dir)

            # Increment random seed for the next trial
            datamodule.random_seed = datamodule.random_seed + 1
    
    # Aggregate across folds and trials to make plotting easier
    aggregate_predictions(config["save_dir"], overwrite=True)
    aggregate_effects(config["save_dir"], overwrite=True)