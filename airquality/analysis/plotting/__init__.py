import matplotlib.pyplot as plt
import yaml
import os

dirpath = os.path.dirname(os.path.abspath(__file__))

plt.style.use(f'{dirpath}/plot_style.mplstyle')

with open(f'{dirpath}/labels_map.yaml', 'r') as f:
    labels_map = yaml.safe_load(f)

from .risk_ratio_plots import plot_effect
from .prediction_plots import plot_aggregate_predictions