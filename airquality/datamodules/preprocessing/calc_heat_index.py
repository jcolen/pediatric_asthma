import numpy as np
import pandas as pd


def calc_heat_index(weather_df, temp_column='AverageDryBulbTemperature', rh_column='AverageRelativeHumidity', convert_fahrenheit=True):
    """Heat Index calculation following https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml """

    RH = weather_df[rh_column]
    if convert_fahrenheit: #LCDv2 files are in SI/metric units
        T = weather_df[temp_column] * 1.8 + 32
    else:
        T = weather_df[temp_column]

    # Simple regression equation first
    heat_index = 0.5 * ( T + 61.0 + ((T-68.0)*1.2) + (RH*0.094) )

    # Compute average with temperature and check if regression equation is needed
    heat_temp_avg = 0.5 * (heat_index + T)
    reg_mask = heat_temp_avg >= 80

    # Apply regresion equation
    heat_index[reg_mask] =  -42.379 + 2.04901523*T[reg_mask] + 10.14333127*RH[reg_mask] \
        - .22475541*T[reg_mask]*RH[reg_mask] - .00683783*T[reg_mask]*T[reg_mask] - .05481717*RH[reg_mask]*RH[reg_mask] \
        + .00122874*T[reg_mask]*T[reg_mask]*RH[reg_mask] + .00085282*T[reg_mask]*RH[reg_mask]*RH[reg_mask] \
        - .00000199*T[reg_mask]*T[reg_mask]*RH[reg_mask]*RH[reg_mask]
    
    # First adjustment
    mask = (reg_mask) & (RH < 13) & (T >= 80) & (T <= 112)
    heat_index[mask] += ( (13-RH[mask]) / 4 )  * np.sqrt( (17 - np.abs(T[mask]-95.)) / 17 )

    # Second adjustment
    mask = (reg_mask) & (RH > 85) & (T >= 80) & (T <= 87)
    heat_index[mask] += ( (RH[mask] - 85) / 10 ) * ( (87 - T[mask]) / 5)
    
    return heat_index