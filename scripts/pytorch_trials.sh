#!/bin/bash

socio_vars=("svi" "coi")
socio_vars=("svi")
n_trials=5

lag="7"
config_updates="datamodule.preprocessing_config.rolling=True datamodule.preprocessing_config.lag=${lag}"

for socio in ${socio_vars[@]}; do
    datamodule_id="pytorch_${socio}_daily"

    python run_pytorch.py \
        --n_trials $n_trials \
        --datamodule_id $datamodule_id \
        --config_updates $config_updates \
        --save_dir $MODELS_DIR/$socio/FCNN/lag${lag}

done