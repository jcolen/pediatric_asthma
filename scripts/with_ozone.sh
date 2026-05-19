#!/bin/bash

socio_vars=("coi")
n_trials=5
lag="7"
lag_updates="datamodule.preprocessing_config.rolling=True datamodule.preprocessing_config.lag=${lag}"

exog="['NO2','SO2','O3','CO','PM25','PM10','COI_overall_perc','AverageDryBulbTemperature','AverageRelativeHumidity','TotalPrecipitation','log_covid']"


for socio in ${socio_vars[@]}; do
    datamodule_id="airquality_${socio}_daily"

    # GLM
    python run_sklearn.py \
        --n_trials $n_trials \
        --datamodule_id $datamodule_id \
        --model_id glm \
        --config_updates $lag_updates datamodule.exog=$exog \
        --save_dir $MODELS_DIR/$socio/GLM/ozone/base

    # Sparse GLM
    # python run_sklearn.py \
    #     --n_trials $n_trials \
    #     --datamodule_id $datamodule_id \
    #     --model_id glm_stlsq \
    #     --config_updates $lag_updates datamodule.exog=$exog \
    #         model.optimizer.threshold=8.5 \
    #     --save_dir $MODELS_DIR/$socio/GLM/ozone/stlsq

done
