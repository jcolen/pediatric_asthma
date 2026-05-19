#!/bin/bash

alerts_file="smoke_alerts_urls.txt"
alerts_dir="smoke_alerts"

mkdir -p $alerts_dir

while read url; do
    filename=$(basename $url)
    wget -O $alerts_dir/$filename $url
    unzip $alerts_dir/$filename -d $alerts_dir
done < $alerts_file