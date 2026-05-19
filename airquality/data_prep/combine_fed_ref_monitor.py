# Merge smoke alerts into a single geopandas dataframe
# Move this to a python file
from shapely.geometry import box
from tqdm.auto import tqdm
import glob
import geopandas as gpd
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # Starting geographic region
    all_fed_ref_monitor = []

    for file in tqdm(sorted(glob.glob('../airquality_toyproblem/data/fed_ref_monitor/*.csv'))):
        df = pd.read_csv(file)
        all_fed_ref_monitor.append(pd.read_csv(file))

    all_fed_ref_monitor = pd.concat(all_fed_ref_monitor, axis=0, ignore_index=True)
    print(all_fed_ref_monitor.head())

    all_fed_ref_monitor.to_csv('../airquality_toyproblem/data/all_fed_ref_monitor.csv', index=False)