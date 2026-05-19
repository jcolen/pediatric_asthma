#!/bin/bash

lag_vars=("1" "3" "5" "7")
socio_vars=("coi") # Just to COI for now, that'll be the primary variable here
n_trials=5

for lag in ${lag_vars[@]}; do
    config_updates="datamodule.preprocessing_config.rolling=True datamodule.preprocessing_config.lag=${lag}"

    for socio in ${socio_vars[@]}; do
        datamodule_id="pytorch_${socio}_daily"

        python run_pytorch.py \
            --n_trials $n_trials \
            --datamodule_id $datamodule_id \
            --config_update $config_updates \
            --save_dir $MODELS_DIR/$socio/FCNN/lag${lag}
    done
done