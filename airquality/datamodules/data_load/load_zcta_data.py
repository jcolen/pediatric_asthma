import pandas as pd
import geopandas as gpd
import numpy as np

def load_zcta_data(path='/Users/jcolen/Documents/chkd_toy_problem/data/Hampton_Roads_ZCTA_Populations/Hampton_Roads_ZCTA_Populations.shp'):
    zcta_df = gpd.read_file(path)
    zcta_df = zcta_df.rename({
        'Total Popu': 'Total Population',
        'Population': 'Population 0-19',
    }, axis=1)
    zcta_df = zcta_df.set_index('ZCTA5CE20')
    return zcta_df

def load_svi_data(path='/Users/jcolen/Documents/chkd_toy_problem/data/Virginia_SVI_ZCTA_2022.csv'):
    svi_df = pd.read_csv(path)
    svi_df['ZCTA5CE20'] = svi_df['LOCATION'].str.replace('ZCTA5 ', '')
    svi_df = svi_df.set_index('ZCTA5CE20')
    svi_df = pd.concat([
        svi_df.filter(like='RPL').rename({
            'RPL_THEME1': 'SVI_socioeco_perc', # Socioeconomic status
            'RPL_THEME2': 'SVI_household_perc', # Household characteristics
            'RPL_THEME3': 'SVI_racial_ethnic_perc', # Racial & ethnic minority status
            'RPL_THEME4': 'SVI_housing_perc', # Housing type & transportation
            'RPL_THEMES': 'SVI_overall_perc', # Overall summary ranking variable (percentile)
        }, axis=1),
        svi_df.filter(like='EP')
    ], axis=1)
    svi_df = svi_df.replace(-999., np.nan).dropna() # Remove null or no data values
    return svi_df

def load_coi_data(path='/Users/jcolen/Documents/chkd_toy_problem/data/COI_ZipCode/data.csv',
                  pop_path='/Users/jcolen/Documents/chkd_toy_problem/data/COI_Population_Estimates/data.csv',
                  year_min=2018, year_max=2023):
    coi_df = pd.read_csv(path, usecols=['zip', 'year', 'r_COI_nat', 'r_ED_nat', 'r_HE_nat', 'r_SE_nat'])
    coi_df = coi_df[(coi_df['year'] >= year_min) & (coi_df['year'] <= year_max)]
    coi_df['ZCTA5CE20'] = coi_df['zip'].astype(str).str.zfill(5)
    coi_df = coi_df.drop(columns='zip') \
        .rename(columns={
            'r_COI_nat': 'COI_overall_perc',
            'r_ED_nat': 'COI_education_perc',
            'r_HE_nat': 'COI_health_env_perc',
            'r_SE_nat': 'COI_socioeco_perc'
        }) \
        .set_index(['year', 'ZCTA5CE20'])
    
    coi_pop = pd.read_csv(pop_path, usecols=['zip', 'year', 'total'])
    coi_pop = coi_pop[(coi_pop['year'] >= 2018) & (coi_pop['year'] <= 2023)]
    coi_pop['ZCTA5CE20'] = coi_pop['zip'].astype(str).str.zfill(5)
    coi_pop = coi_pop.drop(columns='zip'). \
        set_index(['year', 'ZCTA5CE20'])
    
    coi_df['population'] = coi_pop['total']
            
    return coi_df


