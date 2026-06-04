import cv2
import os
import numpy as np

# Path to your dataset folder
DATASET_PATH = "C:\\Users\\LENOVO\\Desktop\\ai project\\dataset"

# Standard image size for FER dataset
IMG_SIZE = 48  

def load_dataset(folder):
    """
    Load images and labels from a specific folder (train/test).

    Returns:
        images: NumPy array of shape (num_images, 48, 48)
        labels: NumPy array of corresponding labels
    """
    images = []
    labels = []

    # Map emotions to numeric labels
    emotion_map = {
        "angry": 0,
        "happy": 1,
        "neutral": 2,
        "sad": 3,
        "surprise": 4
    }

    for emotion, label in emotion_map.items():
        path = os.path.join(DATASET_PATH, folder, emotion)

        # Loop through each image file in the folder
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)

            # Read image as grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to 48x48
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)

def get_data():
    """
    Load training and testing datasets.
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    X_train, y_train = load_dataset("train")
    X_test, y_test = load_dataset("test")
    return X_train, y_train, X_test, y_test

# If this file is run directly, print dataset shapes
if __name__ == "__main__":
    print("Loading training data...")
    X_train, y_train = load_dataset("train")

    print("Loading testing data...")
    X_test, y_test = load_dataset("test")

    print("Shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)
