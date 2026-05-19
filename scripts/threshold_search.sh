#!/bin/bash

thresholds=($(python -c "
import numpy as np; 
t = np.concatenate([np.geomspace(1.001e-2, 50, num=20), np.geomspace(10, 50, num=6)]); 
t = sorted(np.unique(t)); 
print(' '.join(map(str, t)))"))

socio_vars=("svi" "coi")
lag="7"


for t in ${thresholds[@]}; do 
    echo "Running tau = $t"

    for socio in ${socio_vars[@]}; do
        echo "Running socio = $socio"

        echo "Running GLM"
        python run_sklearn.py \
            --n_trials 1 \
            --datamodule_id airquality_${socio}_daily \
            --model_id glm_stlsq \
            --save_dir $MODELS_DIR/${socio}/GLM/lag${lag}/stlsq_threshold/tau=${t} \
            --config_updates \
                datamodule.preprocessing_config.rolling=True \
                datamodule.preprocessing_config.lag=${lag} \
                model.optimizer.threshold=$t \
                datamodule.random_seed=1234
    done
done