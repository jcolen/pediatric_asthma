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

The necessary scripts to run the analyses are located in the `scripts` directory. 
This assumes installation of the required packages from `environment.yml` and that the package has been installed via `pip`.
See `airquality/data_sources.md` for information about all publicly-accessible datasets used in this project.
Children's hospital data on pediatric asthma exacerbations is restricted by Institutional Review Board requirements. 
