import argparse
from collections import namedtuple
from datetime import date, timedelta

import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt

import numpy as np
import scipy.sparse as sp

from feature_extractors import ExtractGPS1DBins
from feature_extractors import ExtractMonth
from feature_extractors import ExtractWeekOfYear
from io_tools import read_data
from model import linear_regression_v1
from model import split_dataset
from model import validate_date

PredictionRecord = namedtuple("PredictionRecord",
                              ["date",
                               "latitude",
                               "longitude",
                               "avg_temp"])

def create_input_records(start_date, end_date, template):
    n_days = (end_date - start_date).days
    records = []
    for i in xrange(n_days):
        d = end_date + timedelta(i)
        record = PredictionRecord(date=d,
                                  latitude=template.latitude,
                                  longitude=template.longitude,
                                  avg_temp=None)
        record.append(record)
    return records

def make_predictions(model, extractors, input_records):
    features = sp.hstack(map(lambda e: e.transform(input_records),
                             extractors))
    avg_temps = model.predict(features)

    output_records = []
    for record, avg_temp in zip(input_records, avg_temps):
        record = record._replace(avg_temp=avg_temp)
        output_records.append(record)

    return output_records

def plot_real_vs_prediction(flname, real, predictions):
    plt.plot(xrange(len(real)),
             real,
             "b.-",
             label="real")
    plt.plot(xrange(len(predictions)),
             predictions,
             "r.-",
             label="predicted")

    plt.legend(loc="best",
               fontsize=16)
    plt.xlabel("Date",
               fontsize=16)
    plt.ylabel("Temperature (F)",
               fontsize=16)
    plt.savefig(flname,
                DPI=300)

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir",
                        type=str,
                        default="data")

    parser.add_argument("--split-date",
                        required=True,
                        type=validate_date)

    parser.add_argument("--wban-station",
                        default="14839",
                        type=str)

    parser.add_argument("--plot-fl",
                        required=True,
                        type=str)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()
    
    print "Reading data"
    records = read_data(args.data_dir)

    print "Splitting data"
    training, testing = split_dataset(records, args.split_date)
    print len(training), "training records"
    print len(testing), "testing records"

    print "Training and Evaluating Model"
    model, extractors = linear_regression_v1(training,
                                             testing,
                                             [ExtractGPS1DBins(n_bins=20),
                                              ExtractMonth(),
                                              ExtractWeekOfYear()])

    print "Building input records"
    real_records = filter(lambda r: r.wban == args.wban_station,
                          records)

    print "Predicting"
    predicted_records = make_predictions(model,
                                         extractors,
                                         real_records)

    print "Plotting"
    real_temperatures = map(lambda r: r.avg_temp,
                            real_records)
    predicted_temperatures = map(lambda r: r.avg_temp,
                                 predicted_records)

    plot_real_vs_prediction(args.plot_fl,
                            real_temperatures,
                            predicted_temperatures)
    
    

    
