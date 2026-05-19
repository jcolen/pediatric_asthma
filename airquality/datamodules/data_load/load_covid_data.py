import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_covid_data(path='/Users/jcolen/Documents/chkd_toy_problem/data/covid_lab_testing.csv'):
    covid_df = pd.read_csv(path)

    # Restrict to positive tests in Virginia
    covid_df = covid_df[covid_df['state'] == 'VA']
    covid_df = covid_df[covid_df['overall_outcome'] == 'Positive']

    covid_df['date_local'] = pd.to_datetime(covid_df['date']).dt.to_period('D')
    covid_df = covid_df.set_index('date_local')[['new_results_reported']]
    covid_df = covid_df.rename({'new_results_reported': 'positive_covid_tests'}, axis=1)

    return covid_df

def plot_covid_tests(ax, covid_df):
    ax.plot(covid_df.index.to_timestamp(), covid_df['positive_covid_tests'])
    ax.set_title('COVID-19 Positive Tests')