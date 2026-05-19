#!/bin/bash

noaa_dir="/Users/jcolen/Documents/chkd_toy_problem/data/NOAA_local_climatological_data_v2"

years=("2018" "2019" "2020" "2021" "2022" "2023")
stations=( \
 "03739" # Cape Charles 5 ENE
 "93735" # Fort Eustis Felker Army Air Field
 "13763" # Franklin Municipal John Beverly Rose Airport
 "03734" # Middle Peninsula Regional Airport
 "93741" # Newport News International Airport
 "03701" # Norfolk Chesapeake Regional Airport
 "13737" # Norfolk International Airport
 "13750" # Norfolk NAS
 "13769" # Oceana NAS
 "03719" # Suffolk Municipal Airport
 "13762" # Fentress Naval Auxiliary Field
 "00154" # Hampton Roads Executive Airport
 "13702" # Langley Air Force Base
 "93773" # Wakefield Municipal Airport
)

for station in ${stations[@]}; do
    for year in ${years[@]}; do
        url="https://www.ncei.noaa.gov/oa/local-climatological-data/v2/access/${year}/LCD_USW000${station}_${year}.csv"
        echo $station $year $url;
        wget -P $noaa_dir/ $url;
    done
done