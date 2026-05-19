#!/bin/bash

socio_vars=("svi" "coi")
n_trials=5

for socio in ${socio_vars[@]}; do
    datamodule_id="airquality_${socio}_daily"

    ### GLM - Single Variable Selection
    python run_sklearn.py \
        --n_trials $n_trials \
        --datamodule_id $datamodule_id \
        --model_id glm_single_variable \
        --save_dir $MODELS_DIR/$socio/GLM/single_variable

    ### GLM - Regularization
    python run_sklearn.py \
        --n_trials $n_trials \
        --datamodule_id $datamodule_id \
        --model_id glm_reg \
        --save_dir $MODELS_DIR/$socio/GLM/regularized

    ### Sparse GLM - Regularization
    python run_sklearn.py \
        --n_trials $n_trials \
        --datamodule_id $datamodule_id \
        --model_id glm_stlsq_reg \
        --save_dir $MODELS_DIR/$socio/GLM/stlsq_regularized

done