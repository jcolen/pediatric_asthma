# Pediatric Asthma Exacerbation code

Code supporting manuscript on modeling pediatric asthma exacerbation from ambient air pollution and other factors.


## Directory Organization
```
├── setup.py
├── airquality
    ├── analysis                          : analysis and plotting modules
    ├── data_prep                         : scripts to clean, combine, and prepare data
    ├── datamodule                        : data loading, preprocessing, and serving
    ├── models                            : predictive models
    ├── utils                             : utilities (backends, optimizers, etc.)
    ├── data_sources.md                   : description of external data files and sources
├── notebooks                             : one-off notebooks for plot generation, etc.
├── scripts                               : pipelines and scripts to run experiments
├── environment.yml                       : conda environment specification
├── setup.py                              : local install script
```

# How to run the code

The basic commands to run the GLM and Sparse models are below. 
Additional scripts are available in the `scripts` folder. 
This assumes that you have installed the requisite packages from `environment.yml` and run `pip install .`.

### GLM - basic

```
python run_sklearn.py \
    --n_trials 5 \
    --datamodule_id airquality_svi_daily \
    --model_id glm \
    --save_dir $MODELS_DIR/svi/GLM/base

python run_sklearn.py \
    --n_trials 5 \
    --datamodule_id airquality_coi_daily \
    --model_id glm \
    --save_dir $MODELS_DIR/coi/GLM/base
```

### GLM - STLSQ

```
python run_sklearn.py \
    --n_trials 5 \
    --datamodule_id airquality_svi_daily \
    --model_id glm_stlsq \
    --save_dir $MODELS_DIR/svi/GLM/stlsq

python run_sklearn.py \
    --n_trials 5 \
    --datamodule_id airquality_coi_daily \
    --model_id glm_stlsq \
    --save_dir $MODELS_DIR/coi/GLM/stlsq
```