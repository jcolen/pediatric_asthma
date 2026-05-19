# List of datasets and sources used in this project

- `Toy Project Outgoing.xlsx/csv` - spreadsheet of patient visits to CHKD sites during the study period
- `fed_ref_monitor` - air quality measurements from federal reference monitors in the Hampton Roads region from 2018-2023. Collected using D. McSpadden's fed_ref_monitor script
- `df_fed_ref_all_monitors.csv` - combined federal reference monitor data from D. McSpadden, April-October from each year
- `purpleair_all.csv` - combined purple air measurements from D. McSpadden, April-October from each year
- `HamptonRoads_2010_Census_Blocks` - shapefiles for Hampton Roads census blocks, acquired from [this link](https://www.hrgeo.org/datasets/HRPDC-GIS::hampton-roads-2010-census-block-groups/explore?location=37.056193%2C-76.667100%2C9.13)
- `smoke_alerts` - regional smoke measurements from NOAA Hazard Mapping System. Acquired using the `get_smoke_alerts.sh` script calling `smoke_alerts_urls.txt`
- `combined_smoke_alerts` - combined regional smoke measurements for each day, merging distinct measurements into non-overlapping polygons using `combine_smoke_alerts.py`
- `tl_2020_us_zcta520` - US Census Zip Code Tabulation Areas, from [this link](https://www.census.gov/programs-surveys/geography/guidance/geo-areas/zctas.html)
- `tl_2024_us_state` - US state boundaries, from [this link](https://www2.census.gov/geo/tiger/TIGER2024/STATE/)
- `chkd_site_addresses.csv` - Manually aggregated geographic locations for each CHKD sites. Addresses identified using CHKD website, lat-lon coordinates obtained by geopandas geocoding functionality
- `chkd_sites` - locations of CHKD sites, obtained using `chkd_sites.ipynb`
- `chkd_air_quality` - air quality measurements at specific CHKD sites, performed using `chkd_geospatial_data_prep.ipynb`
- `chkd_smoke_alerts` - smoke alerts at specific CHKD sites, performed using `chkd_geospatial_data_prep.ipynb`
- `DECENNIALDP2020.DP1_2025-01-27T131504` - 2020 US Census tabulation of populations at the ZCTA level
- `zcta_{param}.csv` - air quality measurements interpolated to ZCTA centroids using `zcta_geospatial_data_prep.ipynb`
- `Virginia_SVI_ZCTA.csv` - Social vulnerability index at Virginia ZCTAs in 2022 (only year available), acquired from [this link](https://www.atsdr.cdc.gov/place-health/php/svi/svi-data-documentation-download.html). Documentation available [here](https://svi.cdc.gov/map25/data/docs/SVI2022Documentation_ZCTA.pdf)
- `NOAA_local_climatological_data_v2` - Historical weather data from NOAA's Local Climatological Data v2 Service [(link)](https://www.ncei.noaa.gov/products/land-based-station/local-climatological-data), downloaded using `get_NOAA_lcd.sh`
- `weekly_covid_hospitalizations.csv` - Weekly lab-confirmed COVID-19 hopsitalizations from the COVID-NET Surveillance System [(link)](https://data.cdc.gov/Public-Health-Surveillance/Weekly-Rates-of-Laboratory-Confirmed-COVID-19-Hosp/6jg4-xsqq/about_data)
- `weekly_covid_cases_deaths.csv` - Weekly United States COVID-19 Cases and Deaths by State [(link)](https://data.cdc.gov/Case-Surveillance/Weekly-United-States-COVID-19-Cases-and-Deaths-by-/pwn4-m3yp/about_data)
- `covid_lab_testing.csv` - COVID-19 Diagnostic Laboratory Testing (PCR Testing) Time Series [(link)](https://healthdata.gov/dataset/COVID-19-Diagnostic-Laboratory-Testing-PCR-Testing/j8mb-icvb/about_data)
- `COI_ZipCode` - Child Opportunity Index (COI) at zip code level from 2012-2023 [(link)](https://www.diversitydatakids.org/research-library/child-opportunity-index-30-2023-zip-code-data)
- `COI_Population_Estimates` - Yearly zip-code level population estimates for ages 0-17 from COI site [(link)](https://www.diversitydatakids.org/research-library/child-opportunity-index-30-2023-zip-code-data)
- `US_Vessel_Traffic_2018_05.gdb` - One month dataset of local maritime vessel traffic [(link)](https://livingatlas.arcgis.com/vessel-traffic/#@=-75.828,36.819,10&time=201805&sublayer=Other)

# Table if ICD10 code frequencies and meanings

| ICD-10 Diagnosis Code | Patient Count | Meaning |
| --------------------- | ------------- | ------- | 
| R05        |  17756 | Cough |
| R05.1      |   6144 | Acute cough|
| R05.2      |    649 | Subacute cough|
| R05.3      |   1493 | Chronic cough |
| R05.8      |    595 | Other specified cough |
| R05.9      |  12390 | Cough, unspecified |
| --------------------- | ------------- | ------- | 
| J06.9      | 126244 | Acute upper respiratory infection, unspecified |
| --------------------- | ------------- | ------- | 
| J45.21     |   1117 | Mild intermittent asthma with (acute) exacerbation |
| J45.31     |   1627 | Mild persistent asthma with (acute) exacerbation |
| J45.41     |   1213 | Moderate persistent asthma with (acute) exacerbation |
| J45.51     |    671 | Severe persistent asthma with (acute) exacerbation |
| J45.901    |   9122 | Unspecified asthma with (acute) exacerbation |