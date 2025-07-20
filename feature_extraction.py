import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

def extract_features(image_path):
    """
    Extracts handcrafted features from an image:
    - Color histograms (RGB)
    - Local Binary Patterns (LBP)
    - Histogram of Oriented Gradients (HOG)
    
    Returns:
        A 1D numpy array of concatenated features.
    """
    # Load image and resize
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded.")
    img = cv2.resize(img, (128, 128))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Color Histogram (RGB)
    hist_r = cv2.calcHist([img], [0], None, [32], [0, 256]).flatten()
    hist_g = cv2.calcHist([img], [1], None, [32], [0, 256]).flatten()
    hist_b = cv2.calcHist([img], [2], None, [32], [0, 256]).flatten()

    # Normalize color histograms
    hist_r = hist_r / (np.sum(hist_r) + 1e-7)
    hist_g = hist_g / (np.sum(hist_g) + 1e-7)
    hist_b = hist_b / (np.sum(hist_b) + 1e-7)

    # 2. Local Binary Pattern (Texture)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59), range=(0, 58))
    lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-7)

    # 3. HOG Features (Edges + Structure)
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )

    # Combine All Features
    feature_vector = np.concatenate([hist_r, hist_g, hist_b, lbp_hist, hog_features])
    return feature_vector


def extract_features_from_array(img):
    """
    Same as extract_features(), but works directly on an image array (e.g., from webcam or upload).
    """
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist_r = cv2.calcHist([img], [0], None, [32], [0, 256]).flatten()
    hist_g = cv2.calcHist([img], [1], None, [32], [0, 256]).flatten()
    hist_b = cv2.calcHist([img], [2], None, [32], [0, 256]).flatten()

    hist_r = hist_r / (np.sum(hist_r) + 1e-7)
    hist_g = hist_g / (np.sum(hist_g) + 1e-7)
    hist_b = hist_b / (np.sum(hist_b) + 1e-7)

    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59), range=(0, 58))
    lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-7)

    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)

    feature_vector = np.concatenate([hist_r, hist_g, hist_b, lbp_hist, hog_features])
    return feature_vector
