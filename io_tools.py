from collections import namedtuple
from datetime import datetime
import glob

Station = namedtuple("Station",
                     ["wban",
                      "latitude",
                      "longitude"])

WeatherRecord = namedtuple("WeatherRecord",
                           ["wban",
                            "date",
                            "latitude",
                            "longitude",
                            "avg_temp",
                            "snowfall",
                            "rainfall",
                            "total_precip",
                            "avg_wind_speed"])

def read_stations(flname):
    fl = open(flname)
    next(fl)
    stations = dict()
    for ln in fl:
        cols = ln.strip().split("|")
        wban = cols[0]
        latitude = float(cols[9])
        longitude = float(cols[10])
        stations[wban] = Station(wban=wban,
                                 latitude=latitude,
                                 longitude=longitude)
    return stations

def read_daily(flname, stations, records):
    fl = open(flname)
    next(fl)
    date_format = "%Y%m%d"
    for ln in fl:
        cols = ln.strip().split(",")
        wban = cols[0]
        if wban not in stations:
            continue
        station = stations[wban]
        date = datetime.strptime(cols[1], date_format).date()
        #temp_max = cols[2]
        #temp_min = cols[4]
        avg_temp = cols[6]

        try:
            snowfall = float(cols[28])
        except:
            snowfall = 0.0

        try:
            rainfall = float(cols[30])
        except:
            rainfall = 0.0

        total_precip = 0.1 * snowfall + rainfall
        
        try:
            avg_wind_speed = float(cols[40])
        except:
            avg_wind_speed = 0.0

        try:
            temp_avg = float(avg_temp)
        except:
            continue

        record = WeatherRecord(wban=wban,
                               date=date,
                               avg_temp=avg_temp,
                               snowfall=snowfall,
                               rainfall=rainfall,
                               total_precip=total_precip,
                               avg_wind_speed=avg_wind_speed,
                               latitude=station.latitude,
                               longitude=station.longitude)
        records.append(record)
    fl.close()

def read_data(data_dir):
    stations = read_stations(data_dir + "/station.txt")
    records = []
    for flname in glob.glob(data_dir + "/*daily.txt"):
        read_daily(flname, stations, records)
    records.sort(key=lambda r: r.date)

    return records
