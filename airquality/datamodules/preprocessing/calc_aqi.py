import numpy as np
import pandas as pd

"""
AQI calculation following https://document.airnow.gov/technical-assistance-document-for-the-reporting-of-daily-air-quailty.pdf
"""

aqi_breakpoints = np.array([
    [0, 50],
    [51, 100],
    [101, 150],
    [151, 200],
    [201, 300],
    [301, 1e10],
])

ozone_breakpoints = np.array([
    [0.000, 0.054],
    [0.055, 0.070],
    [0.071, 0.085],
    [0.086, 0.105],
    [0.106, 0.200],
    [0.201, np.inf]
])

pm25_breakpoints = np.array([
    [0.0, 9.0],
    [9.1, 35.4],
    [35.5, 55.4],
    [55.5, 125.4],
    [125.5, 225.4],
    [225.5, np.inf]
])

pm10_breakpoints = np.array([
    [0, 54],
    [55, 154],
    [155, 254],
    [255, 354],
    [355, 424],
    [425, np.inf]
])

co_breakpoints = np.array([
    [0.0, 4.4],
    [4.5, 9.4],
    [9.5, 12.4],
    [12.5, 15.4],
    [15.5, 30.4],
    [30.5, np.inf]
])

so2_breakpoints = np.array([
    [0, 35],
    [36, 75],
    [76, 185],
    [186, 304],
    [305, 604],
    [605, np.inf]
])

no2_breakpoints = np.array([
    [0, 53],
    [54, 100],
    [101, 360],
    [361, 649],
    [650, 1249],
    [1250, np.inf]
])

def _calc_aqi_measure(measures, breakpoints):
    # Get breakpoints containing measures
    aqi = np.zeros(len(measures))

    for i in range(aqi_breakpoints.shape[0]):
        mask = np.logical_and(measures >= breakpoints[i, 0], measures <= breakpoints[i, 1])
        vals = aqi[mask]

        aqi[mask] = (aqi_breakpoints[i,1] - aqi_breakpoints[i,0]) / (breakpoints[i,1] - breakpoints[i,0]) * \
                    (measures[mask] - breakpoints[i,0]) + aqi_breakpoints[i,0]

    return np.round(aqi, 0)

def calc_aqi_longnames(air_quality):
    aqi = pd.DataFrame(index=air_quality.index)
    aqi['Ozone'] = _calc_aqi_measure(np.round(air_quality['Ozone'], 3), ozone_breakpoints)
    aqi['PM2.5 - Local Conditions'] = _calc_aqi_measure(np.round(air_quality['PM2.5 - Local Conditions'], 1), pm25_breakpoints)
    aqi['PM10 Total 0-10um STP'] = _calc_aqi_measure(np.round(air_quality['PM10 Total 0-10um STP'], 0), no2_breakpoints)
    aqi['Carbon monoxide'] = _calc_aqi_measure(np.round(air_quality['Carbon monoxide'], 1), co_breakpoints)
    aqi['Sulfur dioxide'] = _calc_aqi_measure(np.round(air_quality['Sulfur dioxide'], 0), so2_breakpoints)
    aqi['Nitrogen dioxide (NO2)'] = _calc_aqi_measure(np.round(air_quality['Nitrogen dioxide (NO2)'], 0), no2_breakpoints)
    aqi['AQI'] = aqi.max(axis=1)

    return aqi['AQI']

def calc_aqi(air_quality):
    aqi = pd.DataFrame(index=air_quality.index)
    aqi['O3'] = _calc_aqi_measure(np.round(air_quality['O3'], 3), ozone_breakpoints)
    aqi['PM25'] = _calc_aqi_measure(np.round(air_quality['PM25'], 1), pm25_breakpoints)
    aqi['PM10'] = _calc_aqi_measure(np.round(air_quality['PM10'], 0), no2_breakpoints)
    aqi['CO'] = _calc_aqi_measure(np.round(air_quality['CO'], 1), co_breakpoints)
    aqi['SO2'] = _calc_aqi_measure(np.round(air_quality['SO2'], 0), so2_breakpoints)
    aqi['NO2'] = _calc_aqi_measure(np.round(air_quality['NO2'], 0), no2_breakpoints)
    aqi['AQI'] = aqi.max(axis=1)

    return aqi['AQI']
