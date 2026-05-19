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
    #zcta_df = gpd.read_file('data/tl_2020_us_zcta520/tl_2020_us_zcta520.shp')
    #zcta_df['ZCTA5CE20'] = zcta_df['ZCTA5CE20'].astype(str).str.zfill(5)
    hampton_roads = gpd.read_file('data/Hampton_Roads_2010_Census_Blocks').dissolve('TRACT')
    print(hampton_roads.crs)
    hampton_roads = hampton_roads.to_crs('EPSG:4269')
    print(hampton_roads.crs)
    geographic_region = hampton_roads.geometry.unary_union

    all_smoke_alerts = gpd.GeoDataFrame()

    for file in tqdm(sorted(glob.glob(f'data/smoke_alerts/hms_smoke*.shp'))):
        daily_alerts = gpd.read_file(file).to_crs(hampton_roads.crs)

        daily_alerts = daily_alerts.clip(geographic_region)

        smoke_boundary = None

        # Merge by Density into multipolygons
        merged = []
        #for key in daily_alerts.Density.unique():
        for key in ['Heavy', 'Medium', 'Light']:
            polys = daily_alerts[daily_alerts['Density'] == key].unary_union

            if smoke_boundary is None:
                smoke_boundary = polys
            else:
                polys = polys.difference(smoke_boundary)
                smoke_boundary = smoke_boundary.union(polys)

            # Subtract poly from boundary

            df = pd.DataFrame({'Density': key}, index=[0])
            merged.append(gpd.GeoDataFrame(df, geometry=[polys]))

        # Add in the "None" row

        merged.append(gpd.GeoDataFrame({'Density': 'None'}, index=[0], geometry=[geographic_region.difference(smoke_boundary)]))
        daily_alerts = pd.concat(merged)

        date_str = file[-12:-4]
        daily_alerts['date_local'] = pd.to_datetime(date_str, format='%Y%m%d').to_period('D')
        daily_alerts = daily_alerts.set_index('date_local')

        all_smoke_alerts = pd.concat([all_smoke_alerts, daily_alerts])

        
    all_smoke_alerts = all_smoke_alerts[all_smoke_alerts['geometry'].apply(lambda x: x.geom_type in ['Polygon', 'MultiPolygon'])]
    all_smoke_alerts = all_smoke_alerts[~all_smoke_alerts.geometry.is_empty]
    all_smoke_alerts.to_file('data/combined_smoke_alerts.shp')
