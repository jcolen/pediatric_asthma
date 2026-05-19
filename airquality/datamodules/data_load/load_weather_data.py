import pandas as pd
import matplotlib.pyplot as plt

from ..preprocessing.calc_heat_index import calc_heat_index


def load_weather_data(path='/Users/jcolen/Documents/chkd_toy_problem/data/zcta_daily_weather.csv'):
    weather_df = pd.read_csv(path, index_col=['date_local', 'ZCTA5CE20'])

    # Get Heat Index in Celsius
    weather_df['HeatIndex'] = (calc_heat_index(weather_df) - 32.) * 5/9.

    # Convert index to date and location
    weather_df.index = weather_df.index.set_levels([
        pd.to_datetime(weather_df.index.levels[0]).to_period('D'), 
        weather_df.index.levels[1].astype(str)
    ])
    
    return weather_df

def plot_average_weather_data(fig, ax, weather_df, rolling=None):
    average_weather = weather_df.groupby('date_local').agg('mean')
    if rolling is not None:
        average_weather = average_weather.rolling(window=3, center=True, min_periods=1).mean()

    for i, param in enumerate(average_weather.columns):
        color=plt.cm.tab10(i)
        ax[0,i].plot(average_weather.index.to_timestamp(), average_weather[param], color=color)
        ax[0,i].set(xticks=[f'{year}-06-21' for year in range(2019, 2024)],
                    xticklabels=[f'{year}' for year in range(2019, 2024)]) 
        ax[0,i].set(xlabel='Date')
        ax[0,i].grid(True)
        ax[0,i].set_title(param, color=color)

        average_weather[param].hist(bins=50, ax=ax[1,i], color=color)

    ax[1,0].set_ylabel('Frequency')
    for a in ax.flatten():
        a.tick_params(which='both', direction='in')