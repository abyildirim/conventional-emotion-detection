import argparse
import os
from PIL import Image
import numpy as np
import pandas as pd
from models.sdm_regressor import SDMRegressor
import pickle

def get_images(images_dir):
    print("Reading the images inside the dataset.")
    images = []
    for image_file in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_file)
        image = Image.open(image_path)
        images.append(image)
    images = np.stack(images)
    return images

def save_model(model, model_save_path):
    with open(model_save_path, 'wb') as model_save_file:
        pickle.dump(model, model_save_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="lfpw", help='Name of the dataset.')
    parser.add_argument('--dataset_dir', type=str, default="./datasets/lfpw/processed", help='Directory of the dataset that the regressor is trained on.')
    parser.add_argument('--num_regressors', type=int, default=5, help='Number of cascaded regressors for each landmark coordinate (different regressors used for x and y coordinates of the landmarks)).')
    parser.add_argument('--num_initial_samples', type=int, default=10, help='Number of landmark points sets created initially for each image in the training phase.')
    parser.add_argument('--sift_patch_size', type=int, default=32, help='Patch size used to extract the sift descriptors of the landmarks.')
    parser.add_argument('--model_save_dir', type=str, default="./saved_models", help='Where the trained model will be saved.')
    parser.add_argument('--pca_explained_variance', type=float, default=0.98, help='Explained variance of PCA components of the SIFT descriptors used for training the regressor models.')
    
    args = parser.parse_args()

    train_data_dir = os.path.join(args.dataset_dir, "train")
    images_dir = os.path.join(train_data_dir, "images")
    landmarks_path = os.path.join(train_data_dir, "landmarks.csv")

    images = get_images(images_dir)
    df_landmarks = pd.read_csv(landmarks_path, index_col=0)

    print("SDM regressor training is started.")
    num_landmark_coordinates = len(df_landmarks.columns)
    sdm_regressor = SDMRegressor(args.num_regressors, args.num_initial_samples, num_landmark_coordinates, args.sift_patch_size, args.pca_explained_variance)
    sdm_regressor.fit(images, df_landmarks)
    model_save_dir = os.path.join(args.model_save_dir, args.dataset)
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f"sdm_landmark_regressor_r{args.num_regressors}_p{args.sift_patch_size}_s{args.num_initial_samples}_pca{args.pca_explained_variance}.model")
    save_model(sdm_regressor, model_save_path)
    print("The model is saved to:", model_save_path)
    print("Training is completed!")
     
if __name__ == "__main__":
    main()