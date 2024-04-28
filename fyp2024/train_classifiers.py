import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
import pickle

# Load the metadata and features data
metadata_path =  '..' + os.sep + 'data' + os.sep +'metadata.csv'
features_path = 'features/features.csv'
metadata_df = pd.read_csv(metadata_path)
features_df = pd.read_csv(features_path)
feature_names = ['assymetry', 'colours', 'dots and globules', 'compactness']

# Correct the 'image_id' in features to match 'img_id' in metadata
features_df['image_id'] = features_df['image_id'].str.replace('.png.jpg', '.png')

# Merge features with metadata on 'image_id'/'img_id'
combined_df = features_df.merge(metadata_df[['img_id', 'diagnostic','patient_id']], left_on='image_id', right_on='img_id', how='left')

# Prepare the dataset
feature_columns = combined_df.columns[:-2]  # Exclude 'image_id' and 'img_id' columns
X = combined_df[feature_names].to_numpy()
y = combined_df['diagnostic'] == 'NEV'  # True for 'NEV', False otherwise
patient_id = combined_df['patient_id']

# Prepare cross-validation
num_folds = 5
group_kfold = GroupKFold(n_splits=num_folds)
group_kfold.get_n_splits(X, y, patient_id)

# Initialize classifiers
classifiers = [
    KNeighborsClassifier(1),
    KNeighborsClassifier(5)
]

# Initialize accuracy storage
acc_val = np.empty([num_folds, len(classifiers)])

# Perform cross-validation
for i, (train_index, val_index) in enumerate(group_kfold.split(X, y, patient_id)):
    x_train, y_train = X[train_index], y[train_index]
    x_val, y_val = X[val_index], y[val_index]
    
    for j, clf in enumerate(classifiers):
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_val)
        acc_val[i, j] = accuracy_score(y_val, y_pred)

# Calculate average accuracy
average_acc = np.mean(acc_val, axis=0)
print(f'Classifier 1 average accuracy={average_acc[0]:.3f}')
print(f'Classifier 2 average accuracy={average_acc[1]:.3f}')

# Select the classifier and train on all data
best_classifier = KNeighborsClassifier(n_neighbors=5)
best_classifier.fit(X, y)

# Save the classifier
filename = 'groupXY_classifier.sav'
pickle.dump(best_classifier, open(filename, 'wb'))