import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle

# Load the metadata and features data
metadata_path = '..' + os.sep + 'data' + os.sep + 'metadata.csv'
features_path = 'features' + os.sep + 'features.csv'
metadata_df = pd.read_csv(metadata_path)
features_df = pd.read_csv(features_path)
feature_names = ['assymetry', 'colours', 'dots and globules', 'compactness']

# Correct the 'image_id' in features to match 'img_id' in metadata
features_df['image_id'] = features_df['image_id'].str.replace('.png.jpg', '.png', regex=False)

# Merge features with metadata on 'image_id'/'img_id'
combined_df = features_df.merge(metadata_df[['img_id', 'diagnostic', 'patient_id']], left_on='image_id', right_on='img_id', how='left')

# Check for any NaNs after the merge and handle them before proceeding.
if combined_df.isnull().values.any():
    raise ValueError("NaN values detected after merge! Check the data integrity.")

# Prepare the dataset
X = combined_df[feature_names].to_numpy()
y = combined_df['diagnostic'].values == 'NEV'
patient_id = combined_df['patient_id'].values

# Prepare cross-validation
num_folds = 5
group_kfold = GroupKFold(n_splits=num_folds)

# Initialize classifiers
classifiers = [
    KNeighborsClassifier(1),
    KNeighborsClassifier(5),
    make_pipeline(StandardScaler(), SVC(probability=True)),
    RandomForestClassifier(n_estimators=100, random_state=42),
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    AdaBoostClassifier(n_estimators=100, random_state=42),
    DecisionTreeClassifier(random_state=42),
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3)),
    GaussianNB(),
    MLPClassifier(max_iter=1000)
]

# Initialize accuracy storage
acc_val = np.empty((num_folds, len(classifiers)))

# Perform cross-validation
for i, (train_index, val_index) in enumerate(group_kfold.split(X, y, patient_id)):
    x_train, y_train = X[train_index], y[train_index]
    x_val, y_val = X[val_index], y[val_index]
    
    for j, clf in enumerate(classifiers):
        # Fit and predict
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_val)
        acc_val[i, j] = accuracy_score(y_val, y_pred)

# Calculate average accuracy for each classifier
average_acc = np.mean(acc_val, axis=0)
for idx, clf in enumerate(classifiers):
    # Print the classifier number, name, and accuracy
    classifier_name = clf.named_steps['svc'].__class__.__name__ if 'pipeline' in str(clf) else clf.__class__.__name__
    print(f'Classifier {idx + 1} ({classifier_name}): average accuracy={average_acc[idx]:.3f}')

# Select the best classifier and train on all data
best_index = np.argmax(average_acc)
best_classifier = classifiers[best_index]
best_classifier.fit(X, y)

# Save the best classifier
filename = 'best_classifier.sav'
pickle.dump(best_classifier, open(filename, 'wb'))