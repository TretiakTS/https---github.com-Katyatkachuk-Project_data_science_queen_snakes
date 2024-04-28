import os
from os.path import exists
import pandas as pd
import numpy as np
import cv2

# Import our own file that has the feature extraction functions
from extract_features import extract_features

#-------------------
# Main script
#-------------------

# Where is the raw data
file_data = '..' + os.sep + 'data' + os.sep + 'metadata.csv'
path_image = '..' + os.sep + 'data' + os.sep + 'images' + os.sep + 'imgs_part_1'
path_mask = '..' + os.sep + 'data' + os.sep + 'images' + os.sep + 'Queen_snakes_masks'    

# Where we will store the features
file_features = 'features/features.xlsx'

# Read meta-data into a Pandas dataframe
df = pd.read_csv(file_data)

# Extract image IDs and labels from the data.
image_id = [id + ".jpg" for id in df['img_id']]
mask_id = [id.replace('.png', '_mask.png') + '.jpg' for id in df['img_id']]
label = np.array(df['diagnostic'])

# Make array to store features and list for valid image IDs
feature_names = ['assymetry', 'colours', 'dots and globules', 'compactness']
num_features = len(feature_names)
features = []
valid_image_ids = []

# Loop through all images (limited by num_images)
num_images = len(image_id)
for i in np.arange(num_images):
    # Define filenames related to this image
    file_image = path_image + os.sep + image_id[i]
    file_image_mask = path_mask + os.sep + mask_id[i]

    # Check if both the image and mask files exist
    if exists(file_image) and exists(file_image_mask):
        # Read the image and mask
        im = cv2.imread(file_image)
        mask = cv2.imread(file_image_mask, cv2.IMREAD_GRAYSCALE)

        # Measure features
        x = extract_features(im, mask)

        # Store in the list we created before
        features.append(x)

        # Keep track of the valid image ID
        valid_image_ids.append(image_id[i])

# Create DataFrame from the features list and add image IDs
df_features = pd.DataFrame(features, columns=feature_names)
df_features['image_id'] = valid_image_ids

# Save the image_id used + features to a file
df_features.to_excel(file_features, index=False)