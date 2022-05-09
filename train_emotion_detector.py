import argparse
import os
from PIL import Image
import numpy as np
import pandas as pd
from models.emotion_detector import EmotionDetector
import pickle

def get_images(images_dir):
    print("Reading the images inside the dataset.")
    images = []
    for image_file in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_file)
        image = Image.open(image_path).convert('L') # L: greyscale image
        images.append(image)
    images = np.stack(images)
    return images

def save_model(model, model_save_path):
    with open(model_save_path, 'wb') as model_save_file:
        pickle.dump(model, model_save_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="ck_plus_setup_1", help='Name of the dataset.')
    parser.add_argument('--classifier', type=str, default="svm", help='Type of the classifier.')
    parser.add_argument('--dataset_dir', type=str, default="./datasets/ck_plus/setup_1/processed", help='Directory of the dataset that the regressor is trained on.')
    parser.add_argument('--model_save_dir', type=str, default="./saved_models", help='Where the trained model will be saved.')
    parser.add_argument('--pca_explained_variance', type=float, default=0.98, help='Explained variance of PCA components of the HOG descriptors used for training the regressor models.')
    parser.add_argument('--model_dir', type=str, default="./saved_models", help='Where the trained models will be loaded and saved.')
    parser.add_argument('--model_name', type=str, default="sdm_landmark_regressor_r5_p32_s10_pca0.98", help='Name of the landmark predictor model in the saved models\' directory.')
    

    # python train_emotion_detector.py --dataset ck_plus_setup_1 --classifier svm --dataset_dir ./datasets/ck_plus/setup_1/processed --model_save_dir ./saved_models --pca_explained_variance 0.98
    # python train_emotion_detector.py --dataset ck_plus_setup_1 --classifier random_forest --dataset_dir ./datasets/ck_plus/setup_1/processed --model_save_dir ./saved_models --pca_explained_variance 0.98
    # python train_emotion_detector.py --dataset ck_plus_setup_2 --classifier svm --dataset_dir ./datasets/ck_plus/setup_2/processed --model_save_dir ./saved_models --pca_explained_variance 0.98
    # python train_emotion_detector.py --dataset ck_plus_setup_2 --classifier random_forest --dataset_dir ./datasets/ck_plus/setup_2/processed --model_save_dir ./saved_models --pca_explained_variance 0.98

    args = parser.parse_args()

    train_data_dir = os.path.join(args.dataset_dir, "train")
    images_dir = os.path.join(train_data_dir, "images")
    emotions_path = os.path.join(train_data_dir, "emotions.csv")
    landmark_detector_load_path= os.path.join(args.model_dir, args.dataset, f"{args.model_name}.model")
    print("Loading the landmark detector model.")
    with open(landmark_detector_load_path, 'rb') as model_file:
        landmark_detector = pickle.load(model_file)

    images = get_images(images_dir)
    df_emotions = pd.read_csv(emotions_path, index_col=0)


    print("Emotion detector training is started.")
    emotion_detector = EmotionDetector(args.classifier, landmark_detector, args.pca_explained_variance)
    emotion_detector.fit(images, df_emotions)
    model_save_dir = os.path.join(args.model_save_dir, args.dataset)
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f"emotion_detector_{args.classifier}_pca{args.pca_explained_variance}.model")
    save_model(emotion_detector, model_save_path)
    print("The model is saved to:", model_save_path)
    print("Training is completed!")
     
if __name__ == "__main__":
    main()