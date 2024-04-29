import cv2
import numpy as np
import os
import csv
from sklearn.cluster import KMeans
import pandas as pd
from os.path import exists


def extract_features(image,mask):
    # Initialize an array to store feature values
    num_features=4
    features = np.zeros(num_features, dtype=np.float16)
    
    # Feature 1: Assymetry
    features[0] = check_symmetry(mask)
    
    # Feature 2: Colours
    features[1] = color_analysis(image)
    
    # Feature 3: Dots and Globules
    features[2] = detect_dots(image,mask)
    features[3] = calculate_compactness(mask)
    
    return features


def check_symmetry(img):
    #img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    left_half = img[:, :w//2]
    right_half = img[:, w//2:]
    right_half_flipped = cv2.flip(right_half, 1)
    top_half = img[:h//2, :]
    bottom_half = img[h//2:, :]
    bottom_half_flipped = cv2.flip(bottom_half, 0)
    def is_similar(part_a, part_b, threshold=5):
        if part_a.shape != part_b.shape:
            return False
        diff = cv2.absdiff(np.float32(part_a), np.float32(part_b))
        percent_diff = (np.sum(diff) / (255 * diff.size)) * 100
        return percent_diff < threshold
    horizontal_sym = is_similar(left_half, right_half_flipped)
    vertical_sym = is_similar(top_half, bottom_half_flipped)
    if horizontal_sym and vertical_sym:
        return "1"
    elif horizontal_sym or vertical_sym:
        return "2"
    else:
        return "3"


def process_images_with_masks(image, mask):
    # Create the output directory if it does not exist

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # Apply median blur
    blur = cv2.medianBlur(equalized, 5)

    # Use adaptive thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 25, 8)

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours with area filter
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    larger_dot_area_threshold = 150  # Increase this value to target larger dots
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > larger_dot_area_threshold:
            # Check if contour is approximately circular
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > 0.2:
                # Check if contour touches border
                x, y, w, h = cv2.boundingRect(cnt)
                if x > 1 and y > 1 and (x + w) < image.shape[1] - 1 and (y + h) < image.shape[0] - 1:
                    # Draw circle for each dot
                    cv2.circle(image, (int(x + w / 2), int(y + h / 2)), int((w + h) / 2), (0, 255, 0), 2)

    # Read the mask
    #mask_filename = filename.split('.')[0] + '_mask.png.jpg'  # Assuming naming convention
    #mask_path = os.path.join(masks_directory, mask_filename)
    #mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Resize mask to match image size if they don't match
    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply mask (mask should be 255 for the regions to keep)
    result = cv2.bitwise_and(image, image, mask=mask)

    return result
 

def detect_dots(image,mask):

    image_with_dots=process_images_with_masks(image, mask)
    #Detects green circles in an image.

    #Args:
        #image: The image to process.

    #Returns:
        #1 if a green circle is found, 0 otherwise.

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image_with_dots, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for green color in HSV
    lower_green = np.array([40, 50, 50], dtype="uint8")
    upper_green = np.array([80, 255, 255], dtype="uint8")

    # Create a mask for green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply Hough circle transform to find circles in the mask
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 20,
                                param1=50, param2=30, minRadius=10, maxRadius=200)

    # Check if any circles were found
    if circles is not None:
    # Convert the circles from a tuple to a NumPy array
        circles = np.uint16(np.around(circles[0, :]))

        # Draw the detected circles on the original image (optional)
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)

        # Return 1 if at least one green circle is found
        return 1
    else:
        # Return 0 if no green circles are found
        return 0
    

def calculate_compactness(image):

    # Convert the image to grayscale
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary mask
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the largest contour
    contour = max(contours, key=cv2.contourArea)

    # Calculate area and perimeter of the contour
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Calculate compactness
    compactness = (perimeter ** 2) / (4 * np.pi * area)


    return compactness


def color_analysis(image):
    """
    Analyze the colors present in the input image.

    Args:
    - image (numpy.ndarray): Input image (RGB).

    Returns:
    - num_colors (int): Number of distinct colors present in the image.
    """
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=6, random_state=0)
    kmeans.fit(image.reshape(-1, 3))

    # Filter out similar colors
    _, counts = np.unique(kmeans.labels_, return_counts=True)
    num_colors = len([count for count in counts if count >= 0.05 * image.size])

    return num_colors

def process_images(file_data, path_image, path_mask, feature_names):
    
    # Where we will store the features
    file_features = 'features/features.xlsx'

    # Read meta-data into a Pandas dataframe
    df = pd.read_csv(file_data)

    # Extract image IDs and labels from the data.
    image_id = list(df['img_id'])
    mask_id = [id.replace('.png', '_mask.png') for id in df['img_id']]


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
    return df_features
