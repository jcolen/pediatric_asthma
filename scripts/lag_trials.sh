#!/bin/bash

lag_vars=("0" "1" "3" "5" "7")
socio_vars=("coi") # Just to COI for now, that'll be the primary variable here
socio_vars=("svi") # Sensitivity check on SVI
n_trials=5

for lag in ${lag_vars[@]}; do
    config_updates="datamodule.preprocessing_config.rolling=True datamodule.preprocessing_config.lag=${lag}"

    for socio in ${socio_vars[@]}; do
        datamodule_id="airquality_${socio}_daily"

        # # GLM
        python run_sklearn.py \
            --n_trials $n_trials \
            --datamodule_id $datamodule_id \
            --model_id glm \
            --config_updates $config_updates \
            --save_dir $MODELS_DIR/$socio/GLM/lag${lag}/base

        # Sparse GLM
        python run_sklearn.py \
            --n_trials $n_trials \
            --datamodule_id $datamodule_id \
            --model_id glm_stlsq \
            --config_updates $config_updates model.optimizer.threshold=8.5 \
            --save_dir $MODELS_DIR/$socio/GLM/lag${lag}/stlsq

    done
done