import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_patient_visits(path='/Users/jcolen/Documents/chkd_toy_problem/data/Toy Project Outgoing.csv'):
    patient_df = pd.read_csv(path, low_memory=False, index_col='INDEX_ID')

    # Formatting and typesetting
    patient_df['DATE_OF_SERVICE'] = pd.to_datetime(patient_df['DATE_OF_SERVICE'])
    patient_df['ZCTA5CE20'] = patient_df['PATIENT_ZIPCODE'].astype(str).str.zfill(5)

    # Date columns
    patient_df['date_local'] = patient_df['DATE_OF_SERVICE'].dt.to_period('D')
    patient_df['week_local'] = patient_df['DATE_OF_SERVICE'].dt.to_period('W')

    return patient_df

def plot_counts(ax, patient_df, ICD10_codes=['J06', 'J45', 'R05'], method='daily', rolling=None, scatter=True):
    for i, ICD10_code in enumerate(ICD10_codes):
        icd_df = patient_df[patient_df['QUALIFYING_ICD_10_CODE'].str.startswith(ICD10_code)]

        if method == 'daily':
            counts = icd_df.groupby('date_local').size()
        elif method == 'weekly':
            counts = icd_df.groupby('week_local').size()
        else:
            raise ValueError(f'Argument method={method} unrecognized')
        
        if rolling is not None:
            counts = counts.rolling(window=rolling, min_periods=1).mean()
        
        color = plt.cm.tab10(i)
        if scatter:
            ax.scatter(counts.index.to_timestamp(), counts, s=1, color=color, label=ICD10_code)
        else:
            ax.plot(counts.index.to_timestamp(), counts, color=color, label=ICD10_code)

    return ax