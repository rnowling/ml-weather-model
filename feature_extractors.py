import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import lil_matrix

class ExtractTemperatures(object):
    def fit(self, records):
        pass

    def transform(self, records):
        features = np.zeros((len(records),))
        for i, record in enumerate(records):
            features[i] = record.avg_temp
        return features

class ExtractWeekOfYear(object):
    def fit(self, records):
        pass

    def transform(self, records):
        # per datetime docs, iso calendar has 53 weeks per year
        features = lil_matrix((len(records),
                               53))
        for i, record in enumerate(records):
            week_idx = record.date.isocalendar()[1] - 1
            features[i, week_idx] = 1.

        return features.tocsr()

class ExtractGPSClusters(object):
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, records):
        gps_coordinates = set()
        for record in records:
            coordinates = (record.latitude, \
                           record.longitude)
            gps_coordinates.add(coordinates)


        coordinate_matrix = lil_matrix((len(gps_coordinates),
                                        2))
        for i, (latitude, longitude) in enumerate(gps_coordinates):
            coordinate_matrix[i, 0] = latitude
            coordinate_matrix[i, 1] = longitude

        coordinate_matrix = coordinate_matrix.tocsr()
        self.model = KMeans(n_clusters=self.n_clusters)
        self.model.fit(coordinate_matrix.tocsr())
        print "Inertia", self.model.inertia_

    def transform(self, records):
        coordinates = lil_matrix((len(records),
                                  2))
        for i, record in enumerate(records):
            coordinates[i, 0] = record.latitude
            coordinates[i, 1] = record.longitude

        clusters = lil_matrix((len(records),
                               self.n_clusters))
        for i, cluster_idx in enumerate(self.model.predict(coordinates.tocsr())):
            clusters[i, cluster_idx] = 1.


        return clusters
    
class ExtractMonth(object):
    def fit(self, records):
        pass

    def transform(self, records):
        features = lil_matrix((len(records),
                               12))
        for i, record in enumerate(records):
            month_idx = record.date.month - 1
            features[i, month_idx] = 1.

        return features.tocsr()

class Extract1DBins(object):
    def __init__(self, n_bins, field_name):
        self.n_bins = n_bins
        self.field_name = field_name
    
    def fit(self, records):
        self.min_value = 1e8
        self.max_value = -1e8
        for rec in records:
            value = rec._asdict()[self.field_name]
            self.min_value = min(self.min_value, value)
            self.max_value = max(self.max_value, value)

        self.value_range = self.max_value - self.min_value

    def transform(self, records):
        features = lil_matrix((len(records),
                               self.n_bins))
        for i, rec in enumerate(records):
            value = rec._asdict()[self.field_name]
            bounded = min(value,
                          self.max_value)
            bounded = max(bounded,
                          self.min_value)
            bin_idx = int((bounded - self.min_value) / self.value_range * self.n_bins)
            bin_idx = min(self.n_bins - 1, bin_idx)
            features[i, bin_idx] = 1.0

        return features.tocsr()

class ExtractGPS2DBins(object):
    def __init__(self, n_bins):
        self.n_bins = n_bins
    
    def fit(self, records):
        self.min_lat = 1e8
        self.min_long = 1e8
        max_lat = -1e8
        max_long = -1e8
        for rec in records:
            self.min_lat = min(self.min_lat, rec.latitude)
            max_lat = max(max_lat, rec.latitude)
            self.min_long = min(self.min_long, rec.longitude)
            max_long = max(max_long, rec.longitude)

        self.lat_range = max_lat - self.min_lat
        self.long_range = max_long - self.min_long

    def transform(self, records):
        features = lil_matrix((len(records),
                               self.n_bins * self.n_bins))
        for i, rec in enumerate(records):
            lat_bin = int((rec.latitude - self.min_lat) / self.lat_range * self.n_bins)
            lat_bin = min(self.n_bins - 1, lat_bin)
            long_bin = int((rec.longitude - self.min_long) / self.long_range * self.n_bins)
            long_bin = min(self.n_bins - 1, long_bin)
            gps_bin = lat_bin * self.n_bins + long_bin
            features[i, gps_bin] = 1.0
        return features.tocsr()
