import pickle

# Assuming extract_features.py is in the same directory and has a function named extract_features
from extract_features import extract_features

# The function to classify new images. The image and mask are assumed to be loaded.
def classify(img, mask):
    # Resize the image etc, if you did that during training.
    # ...

    # Extract features (the same ones that you used for training).
    # Ensure that the feature array is 2D: 1 row of features for 1 example.
    x = extract_features(img, mask).reshape(1, -1)

    # Load the trained classifier.
    # Make sure the file name matches the one you used when saving the model.
    classifier = pickle.load(open('best_classifier.sav', 'rb'))

    # Use it on this example to predict the label AND posterior probability.
    pred_label = classifier.predict(x)
    pred_prob = classifier.predict_proba(x)

    # Uncomment below to print the results if needed.
    # print('Predicted label is:', pred_label)
    # print('Predicted probability is:', pred_prob)
    return pred_label, pred_prob

# Note: This code assumes that the function extract_features returns a NumPy array
# that can be reshaped to 1 row for the classifier's predict and predict_proba methods.
# You may need to adjust the reshape depending on the output of your feature extraction.
