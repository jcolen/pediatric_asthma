import pandas as pd
import matplotlib.pyplot as plt

from ..preprocessing.calc_aqi import calc_aqi

def load_air_data(path='/Users/jcolen/Documents/chkd_toy_problem/data'):
    air_df = pd.concat([
        pd.read_csv(f'{path}/zcta_carbon_monoxide.csv', index_col=['date_local', 'ZCTA5CE20']),
        pd.read_csv(f'{path}/zcta_nitrogen_dioxide.csv', index_col=['date_local', 'ZCTA5CE20']),
        pd.read_csv(f'{path}/zcta_sulfur_dioxide.csv', index_col=['date_local', 'ZCTA5CE20']),
        pd.read_csv(f'{path}/zcta_ozone.csv', index_col=['date_local', 'ZCTA5CE20']),
        pd.read_csv(f'{path}/zcta_PM25.csv', index_col=['date_local', 'ZCTA5CE20']),
        pd.read_csv(f'{path}/zcta_PM10.csv', index_col=['date_local', 'ZCTA5CE20']),
        pd.read_csv(f'{path}/zcta_smoke_alerts.csv', index_col=['date_local', 'ZCTA5CE20']).fillna('None'),
    ], axis=1)

    # Categorical data type for smoke density
    density_dtype = pd.api.types.CategoricalDtype(categories=['None', 'Light', 'Medium', 'Heavy'], ordered=True)
    air_df['Smoke'] = air_df['Smoke_Density'].fillna('None').astype(density_dtype).cat.codes
    air_df = air_df.drop('Smoke_Density', axis=1)

    # Get AQI info
    air_df = air_df.rename({'PM2.5': 'PM25'}, axis=1)
    air_df['AQI'] = calc_aqi(air_df)

    # Set index as date and zip code
    air_df.index = air_df.index.set_levels([
        pd.to_datetime(air_df.index.levels[0]).to_period('D'), 
        air_df.index.levels[1].astype(str)
    ])

    return air_df

def plot_average_air_data(fig, ax, air_df, rolling=None):
    average_air = air_df.groupby('date_local').agg('mean')
    if rolling is not None:
        average_air = average_air.rolling(window=rolling, center=True, min_periods=1).mean()

    for i, param in enumerate(average_air.columns):
        color=plt.cm.tab10(i)
        ax[0,i].plot(average_air.index.to_timestamp(), average_air[param], color=color)
        ax[0,i].set(xticks=[f'{year}-06-21' for year in range(2019, 2024)],
                    xticklabels=[f'{year}' for year in range(2019, 2024)]) 
        ax[0,i].set(xlabel='Date')
        ax[0,i].grid(True)
        ax[0,i].set_title(param, color=color)

        average_air[param].hist(bins=50, ax=ax[1,i], color=color)

    ax[1,0].set_ylabel('Frequency')
    for a in ax.flatten():
        a.tick_params(which='both', direction='in')