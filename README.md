# Pediatric Asthma Exacerbation code

Code supporting the preprint: [Learning to model pediatric asthma exacerbation from multiple risk factors: a case study in coastal Virginia](https://arxiv.org/abs/2606.06174), which models pediatric asthma exacerbation from ambient air pollution and other factors.


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

## Citing this repository

If you use this code, please cite

    @misc{colen_pediatric_asthma_2026,
        title = {Learning to model pediatric asthma exacerbation from multiple risk factors: a case study in coastal {Virginia}},
        author = {Colen, Jonathan and Werner, Eric and Golbazi, Maryam and Richter, Heather and McSpadden, Diana and Quinn, Amy and Santos, Jocel and Darling, Mary Jane and Gleason, Mary Margaret},
        year = {2026},
        archivePrefix = {arXiv},
        primaryClass = {cs.LG},
        eprint = {2606.06174},
    }
