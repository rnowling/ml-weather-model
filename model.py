import argparse
from datetime import datetime

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from feature_extractors import Extract1DBins
from feature_extractors import ExtractGPS2DBins
from feature_extractors import ExtractGPSClusters
from feature_extractors import ExtractMonth
from feature_extractors import ExtractTemperatures
from feature_extractors import ExtractWeekOfYear
from io_tools import read_data

def split_dataset(records, cutoff_date):
    cutoff_idx = -1
    for i, record in enumerate(records):
        if cutoff_date < record.date:
            cutoff_idx = i
            break

    return records[:i], records[i:]

def validate_date(s):
    format_str = "%Y-%m-%d"
    try:
        return datetime.strptime(s, format_str).date()
    except ValueError:
        msg = "%s is not a valid date" % s
        raise argparse.ArgumentTypeError(msg)

def linear_regression_v1(training, testing, extractors):
    training_temperatures = ExtractTemperatures().transform(training)
    testing_temperatures = ExtractTemperatures().transform(testing)

    for extractor in extractors:
        extractor.fit(training)

    training_features = sp.hstack(map(lambda e: e.transform(training),
                                      extractors))
    testing_features = sp.hstack(map(lambda e: e.transform(testing),
                                     extractors))

    regressor = SGDRegressor(n_iter=20)
    regressor.fit(training_features,
                  training_temperatures)

    true_costs = testing_temperatures
    pred_costs = regressor.predict(testing_features)
    mae = mean_absolute_error(true_costs, pred_costs)
    mse = mean_squared_error(true_costs, pred_costs)
    r2 = regressor.score(testing_features,
                         true_costs)

    print "MAE:", mae
    print "MSE:", mse
    print "R2:", r2
    print

    return regressor, extractors

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir",
                        type=str,
                        default="data")

    parser.add_argument("--split-date",
                        required=True,
                        type=validate_date)

    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()

    print "Reading data"
    records = read_data(args.data_dir)

    print "Splitting data"
    training, testing = split_dataset(records, args.split_date)
    print len(training), "training records"
    print len(testing), "testing records"

    """
    print "Evaluating months"
    linear_regression_v1(training,
                         testing,
                         [ExtractMonth()])

    print "Evaluating Latitude 1D Bins (bins=10)"
    linear_regression_v1(training,
                         testing,
                         [Extract1DBins(10, "latitude")])
    
    print "Evaluating Latitude 1D Bins (bins=20)"
    linear_regression_v1(training,
                         testing,
                         [Extract1DBins(20, "latitude")])

    print "Evaluating Latitude 1D Bins (bins=30)"
    linear_regression_v1(training,
                         testing,
                         [Extract1DBins(30, "latitude")])

    print "Evaluating Longitude 1D Bins (bins=10)"
    linear_regression_v1(training,
                         testing,
                         [Extract1DBins(10, "longitude")])
    
    print "Evaluating Longitude 1D Bins (bins=20)"
    linear_regression_v1(training,
                         testing,
                         [Extract1DBins(20, "longitude")])

    print "Evaluating Longitude 1D Bins (bins=30)"
    linear_regression_v1(training,
                         testing,
                         [Extract1DBins(30, "longitude")])

    print "Evaluating Longitude 1D Bins (bins=40)"
    linear_regression_v1(training,
                         testing,
                         [Extract1DBins(30, "longitude")])

    print "Evaluating Longitude 1D Bins (bins=50)"
    linear_regression_v1(training,
                         testing,
                         [Extract1DBins(30, "longitude")])
    
    print "Evaluating GPS 2D Bins"
    linear_regression_v1(training,
                         testing,
                         [ExtractGPS2DBins(n_bins=20)])

    print "Evaluate GPS Clusters (n=25)"
    linear_regression_v1(training,
                         testing,
                         [ExtractGPSClusters(n_clusters=25)])

    print "Evaluate GPS Clusters (n=50)"
    linear_regression_v1(training,
                         testing,
                         [ExtractGPSClusters(n_clusters=50)])

    print "Evaluate GPS Clusters (n=75)"
    linear_regression_v1(training,
                         testing,
                         [ExtractGPSClusters(n_clusters=75)])

    print "Evaluate GPS Clusters (n=100)"
    linear_regression_v1(training,
                         testing,
                         [ExtractGPSClusters(n_clusters=100)])

    print "Evaluate GPS Clusters (n=125)"
    linear_regression_v1(training,
                         testing,
                         [ExtractGPSClusters(n_clusters=125)])
    

    print "Evaluating Week of Year"
    linear_regression_v1(training,
                         testing,
                         [ExtractWeekOfYear()])
    """

    print "Evaluating GPS Cluster(n=100), Week Of Year, and months"
    linear_regression_v1(training,
                         testing,
                         [ExtractGPSClusters(n_clusters=100),
                          ExtractWeekOfYear()])

    
    
    
