import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import geopy

import contextily as ctx
import shapely

from geopandas.tools import geocode

if __name__ == '__main__':
    print('Loading CHKD address file')
    chkd_df = pd.read_csv('./chkd_site_addresses.csv', index_col='HEALTH_CARE_SITE')

    print('Geocoding CHKD addresses')
    chkd_gdf = geocode(chkd_df['Address'], provider='nominatim', user_agent='airquality_toyproblem', timeout=4)
    # Manually add CMG-PAW using lat-long point from google maps
    chkd_gdf.loc['CMG-PAW', 'geometry'] = shapely.geometry.point.Point(-76.2905, 36.8533)

    # merge and set address points
    chkd_gdf['Address'] = chkd_df.Address
    chkd_gdf = chkd_gdf.drop(columns='address').reset_index(names='HEALTH_CARE_SITE')

    print('Saving CHKD geocoded addresses')
    chkd_gdf.to_file('chkd_sites/chkd_site_addresses.shp')