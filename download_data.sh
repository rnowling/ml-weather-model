#!/usr/bin/env bash
mkdir -p data/weather
cd data/weather
for year in 1997 1998 1999 2000 2001 2002 2003 2004 2005 2006
do
    for month in 01 02 03 04 05 06 07 08 09 10 11 12
    do
        wget https://www.ncdc.noaa.gov/orders/qclcd/${year}${month}.tar.gz
    done
done
