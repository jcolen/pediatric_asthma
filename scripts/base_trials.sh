#!/bin/bash

socio_vars=("svi" "coi")
n_trials=5

for socio in ${socio_vars[@]}; do
    datamodule_id="airquality_${socio}_daily"

    # GLM
    python run_sklearn.py \
        --n_trials $n_trials \
        --datamodule_id $datamodule_id \
        --model_id glm \
        --save_dir $MODELS_DIR/$socio/GLM/base

    # Sparse GLM
    python run_sklearn.py \
        --n_trials $n_trials \
        --datamodule_id $datamodule_id \
        --model_id glm_stlsq \
        --save_dir $MODELS_DIR/$socio/GLM/stlsq

done