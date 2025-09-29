import joblib
import cv2
from skimage.feature import local_binary_pattern, hog
import numpy as np
from joblib import load

# Load the trained SVM model and feature params
# model = load('gender/models/dros_or_not_model.pkl')
# params = load('gender/models/dros_feature_extraction_params.pkl')

def process_new_image(image):
    model = load('gender/models/dros_or_not_model.pkl')
    params = load('gender/models/dros_feature_extraction_params.pkl')
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to the expected input size
    image = cv2.resize(image, params['preprocessing']['target_size'])
    
    # Preprocessing (CLAHE + Gaussian Blur)
    clahe = cv2.createCLAHE(**params['preprocessing']['clahe_params'])
    image = clahe.apply(image)
    image = cv2.GaussianBlur(image, params['preprocessing']['gaussian_blur'], 0)
    
    # Compute LBP features
    lbp_image = local_binary_pattern(image, **params['lbp_params'])
    n_bins = params['lbp_params']['P'] + 2
    lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=n_bins, range=(0, n_bins))
    lbp_hist = lbp_hist.astype('float32')
    lbp_hist = (lbp_hist + 1e-7) / (np.sum(lbp_hist) + 1e-7)
    
    # Compute HOG features
    hog_features = hog(image, **params['hog_params'])
    
    # Combine features
    combined_features = np.concatenate([lbp_hist, hog_features])
    prediction = model.predict([combined_features])[0]
    return prediction



def load_random_forest(random_forest_model_path, hog_param_path):
    """
    Load a random forest model from the specified path.
    """
    return joblib.load(random_forest_model_path), joblib.load(hog_param_path)


def load_local_binary_pattern(lbp_path, feature_extractor_params_path):
    """
    Load local binary patterns from the specified path.
    """
    return joblib.load(lbp_path), joblib.load(feature_extractor_params_path)

import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog

def read_image(image_path):
    """
    Read an image from the specified path.
    
    Args:
    - image_path: Path to the image file.
    
    Returns:
    - image: The loaded image.
    """
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image was loaded successfully
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    return image

def extract_hog_features(image, hog_params):
    """
    Extract HOG features from the given image.
    
    Args:
    - image: The input image.
    - hog_params: Parameters for the HOG function.
    
    Returns:
    - hog_features: The extracted HOG features.
    """
    # Extract HOG features
    # hog_features, hog_image = hog(
    #     image, 
    #     pixels_per_cell=hog_params['pixels_per_cell'], 
    #     cells_per_block=hog_params['cells_per_block'], 
    #     visualize=True, 
    #     block_norm='L2-Hys'
    # )
    hog_features, hog_image = hog(image, **hog_params, visualize=True)
    
    return hog_features, hog_image

def lbp_preprocess_and_predict(model, image, params, hog_features=None):
    """
    Preprocess the image, extract LBP features, and make a prediction using the model.
    
    Args:
    - model: The trained model for prediction.
    - image: The input image.
    - params: A dictionary containing preprocessing and feature extraction parameters.
    
    Returns:
    - prediction: The predicted class ('Male' or 'Female').
    """
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to the expected input size
    image = cv2.resize(image, params['preprocessing']['target_size'])
    
    # Preprocessing (CLAHE + Gaussian Blur)
    clahe = cv2.createCLAHE(**params['preprocessing']['clahe_params'])
    image = clahe.apply(image)
    image = cv2.GaussianBlur(image, params['preprocessing']['gaussian_blur'], 0)
    
    # Compute LBP features
    lbp_image = local_binary_pattern(image, **params['lbp_params'])
    n_bins = params['lbp_params']['P'] + 2
    lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=n_bins, range=(0, n_bins))
    lbp_hist = lbp_hist.astype('float32')
    lbp_hist = (lbp_hist + 1e-7) / (np.sum(lbp_hist) + 1e-7)
    
    # Compute HOG features using the extract_hog_features function
    # hog_features = extract_hog_features(image, params['hog_params'])
    hog_features = hog(image, **params['hog_params'])
    # Combine features
    combined_features = np.concatenate([lbp_hist, hog_features])
    
    # Prediction
    prediction = model.predict([combined_features])[0]
    prediction_score = model.predict_proba([combined_features])[0]
    # print("hei", prediction2)

    gender = 'Male' if prediction == 0 else 'Female'

    return gender, prediction_score



def rf_preprocess_and_predict(model, hog_features):
    hog_feature = hog_features.reshape(1, -1)
    
    # Predict gender
    prediction = model.predict(hog_feature)
    prediction_score = model.predict_proba(hog_feature)[0]
    # print(prediction2)
    # score = model.predict_proba(hog_feature) 
    gender = 'Male' if prediction[0] == 0 else 'Female'
    
    return gender, prediction_score

import numpy as np

def fuse_probabilities(lbp_pred, lbp_score, rf_score):
    """
    Fuse probability scores from LBP and Random Forest models based on LBP prediction.

    Parameters:
    - lbp_pred (str): Prediction from the LBP model ('Female' or 'Male').
    - lbp_score (list or np.ndarray): Probability scores from the LBP model.
    - rf_score (list or np.ndarray): Probability scores from the Random Forest model.

    Returns:
    - fused_score (np.ndarray): Combined probability scores after applying weights.
    """
    # Ensure inputs are numpy arrays
    lbp_score = np.array(lbp_score)
    rf_score = np.array(rf_score)

    # Assign weights based on LBP prediction
    if lbp_pred == 'Female':
        lbp_weight = 0.7
        rf_weight = 0.3
    else:
        lbp_weight = 0.3
        rf_weight = 0.7

    # Compute the fused probability scores
    fused_score = lbp_weight * lbp_score + rf_weight * rf_score
    print(fused_score)

    return fused_score
