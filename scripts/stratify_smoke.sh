#!/bin/bash

socio_vars=("coi")
n_trials=5

lag="7"
lag_updates="datamodule.preprocessing_config.rolling=True datamodule.preprocessing_config.lag=${lag}"

for socio in ${socio_vars[@]}; do
    datamodule_id="airquality_${socio}_daily"

    python run_sklearn.py \
        --n_trials $n_trials \
        --datamodule_id $datamodule_id \
        --model_id glm \
        --save_dir $MODELS_DIR/$socio/GLM/smoke/none \
        --config_updates \
            $lag_updates \
            datamodule.preprocessing_config.stratify_exposures="{'column': 'Smoke', 'values': [0,0.5]}"

    python run_sklearn.py \
        --n_trials $n_trials \
        --datamodule_id $datamodule_id \
        --model_id glm \
        --save_dir $MODELS_DIR/$socio/GLM/smoke/low \
        --config_updates \
            $lag_updates \
            datamodule.preprocessing_config.stratify_exposures="{'column': 'Smoke', 'values': [0.5,1.5]}"

    python run_sklearn.py \
        --n_trials $n_trials \
        --datamodule_id $datamodule_id \
        --model_id glm \
        --save_dir $MODELS_DIR/$socio/GLM/smoke/med_high \
        --config_updates \
            $lag_updates \
            datamodule.preprocessing_config.stratify_exposures="{'column': 'Smoke', 'values': [1.5, 5.5]}"

    python run_sklearn.py \
        --n_trials $n_trials \
        --datamodule_id $datamodule_id \
        --model_id glm \
        --save_dir $MODELS_DIR/$socio/GLM/smoke/low_med_high \
        --config_updates \
            $lag_updates \
            datamodule.preprocessing_config.stratify_exposures="{'column': 'Smoke', 'values': [0.5, 5.5]}"

done