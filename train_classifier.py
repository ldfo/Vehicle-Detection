import argparse
import numpy as np
import os
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from extract_features import extract_features_from_images
import yaml

def find_images(directory):
    '''function to find images'''
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.split('.')[-1].lower() in ['jpg', 'png']:
                yield os.path.join(root, f)

parser = argparse.ArgumentParser()
parser.add_argument('vehicle_directory', type=str)
parser.add_argument('non_vehicle_directory', type=str)
args = parser.parse_args()

feature_parameters = {
    'cspace': 'YCrCb',
    'spatial_size': (32, 32),
    'hist_bins': 32,
    'hog_orient': 9,
    'hog_pix_per_cell': 8,
    'hog_cell_per_block': 2,
    'hog_channel': 'ALL'
}

print("Loading vehicle images and extracting features...")
vehicle_features = extract_features_from_images(find_images(args.vehicle_directory),
                                                **feature_parameters)

print("Loading non-vehicle images and extracting features...")
non_vehicle_features = extract_features_from_images(find_images(args.non_vehicle_directory),
                                                    **feature_parameters)

print("Extracted features from {} non-vehicle images".format(len(non_vehicle_features)))

X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)
y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print("Training...")
svc = LinearSVC()
svc.fit(X_train, y_train)

positives = y_test.nonzero()
negatives = np.logical_not(y_test).nonzero()
tp = svc.score(X_test[positives], y_test[positives])
tn = svc.score(X_test[negatives], y_test[negatives])
print('True Positive Rate:  {:.2f}%'.format(100*tp))
print('True Negative Rate:  {:.2f}%'.format(100*tn))
print('False Positive Rate: {:.2f}%'.format(100*(1-tn)))
print('False Negative Rate: {:.2f}%'.format(100*(1-tp)))

# print('Pickling classifier to classifier.p')
# with open('classifier.p', 'wb') as f:
#     data = {
#         'feature_parameters': feature_parameters,
#         'classifier': svc,
#         'shape': (64, 64),
#         'scaler': scaler
#     }
#     pickle.dump(data, f)
