import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_effect(ax : plt.Axes,
                risk_df : pd.DataFrame,
                x_col='mean',
                y_col='parameter',
                var_col='variance',
                labels_map : dict = None,
                ticks_map : dict = None,
                label=None,
                color='black',
                offset=0.,
                marker='d',
                **kwargs):
    # Effect plot
    if labels_map is not None:
        risk_df[y_col] = risk_df[y_col].map(labels_map)
        risk_df = risk_df.loc[risk_df[y_col].notnull()].copy()

    y = risk_df[y_col].map(ticks_map) + offset
    x = risk_df[x_col]
    if var_col is None:
        xerr=None
    else: # stdev error bars
        xerr = np.stack([
            np.sqrt(risk_df[var_col]),
            np.sqrt(risk_df[var_col])
        ])

    # Print things for reporting
    tick_labels = list(ticks_map.keys())
    tick_values = np.array([ticks_map[key] for key in ticks_map.keys()])
    for i in range(len(x)):
        print(f'{y.iloc[i]:4.1f}, {tick_labels[np.argwhere(tick_values == int(y.iloc[i]))[0][0]]:30s}: {x.iloc[i]:8.3f} +/- {xerr[0,i]:4.3f}')

    ax.errorbar(
        y=y,
        x=x,
        xerr=xerr,
        linestyle='',
        label=label,
        marker=marker,
        markerfacecolor='white',
        markeredgecolor=color,
        ecolor=color,
        **kwargs,
    )

    ax.set(yticks=[ticks_map[key] for key in ticks_map.keys()], yticklabels=ticks_map.keys())