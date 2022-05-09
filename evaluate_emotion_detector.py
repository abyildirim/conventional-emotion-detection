from PIL import Image
import numpy as np
import pandas as pd
import os
import pandas as pd
import cv2
import pickle
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from utils.triangular_warper import triangular_warp


def get_images(images_dir):
    print("Reading the images inside the dataset.")
    images = []
    for image_file in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_file)
        image = Image.open(image_path).convert('L') # L: greyscale image
        images.append(image)
    images = np.stack(images)
    return images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="ck_plus_setup_1", help='Name of the dataset.')
    parser.add_argument('--dataset_dir', type=str, default="./datasets/ck_plus/setup_1/processed", help='Directory of the dataset that the regressor is trained on.')
    parser.add_argument('--output_dir', type=str, default="./output", help='Where the results will be saved.')
    parser.add_argument('--model_dir', type=str, default="./saved_models", help='Where the trained model will be loaded.')
    parser.add_argument('--model_name', type=str, default="emotion_detector_svm_pca0.98", help='Name of the model in the saved models\' directory.')
    args = parser.parse_args()

    # python evaluate_emotion_detector.py --dataset ck_plus_setup_1 --model_name emotion_detector_svm_pca0.98 --dataset_dir ./datasets/ck_plus/setup_1/processed --output_dir ./output --model_dir ./saved_models
    # python evaluate_emotion_detector.py --dataset ck_plus_setup_1 --model_name emotion_detector_random_forest_pca0.98 --dataset_dir ./datasets/ck_plus/setup_1/processed --output_dir ./output --model_dir ./saved_models
    # python evaluate_emotion_detector.py --dataset ck_plus_setup_2 --model_name emotion_detector_svm_pca0.98 --dataset_dir ./datasets/ck_plus/setup_2/processed --output_dir ./output --model_dir ./saved_models
    # python evaluate_emotion_detector.py --dataset ck_plus_setup_2 --model_name emotion_detector_random_forest_pca0.98 --dataset_dir ./datasets/ck_plus/setup_2/processed --output_dir ./output --model_dir ./saved_models
    
    model_output_dir = os.path.join(args.output_dir, args.dataset, args.model_name)
    os.makedirs(model_output_dir,exist_ok=True)
    model_load_path= os.path.join(args.model_dir, args.dataset, f"{args.model_name}.model")
    test_images_dir = os.path.join(args.dataset_dir, "test", "images")
    test_emotions_path = os.path.join(args.dataset_dir, "test", "emotions.csv")
    images = get_images(test_images_dir)
    df_emotions_test = pd.read_csv(test_emotions_path, index_col=0)

    print("Loading the classifier model.")
    with open(model_load_path, 'rb') as model_file:
        classifier_model = pickle.load(model_file)

    z_fill_length = len(str(len(images)))
    emotions_output_dir = os.path.join(model_output_dir, "emotion_results")
    os.makedirs(emotions_output_dir,exist_ok=True)
    convex_hulls_output_dir = os.path.join(model_output_dir, "convex_hulls")
    os.makedirs(convex_hulls_output_dir,exist_ok=True)
    normalized_images_output_dir = os.path.join(model_output_dir, "normalized_images")
    os.makedirs(normalized_images_output_dir,exist_ok=True)
    triangle_images_output_dir = os.path.join(model_output_dir, "triangle_images")
    os.makedirs(triangle_images_output_dir,exist_ok=True)

    print("Image processing visualization is started.")
    landmarks_list = [classifier_model.landmark_detector.predict(image) for image in tqdm(images,desc='Extracting the landmark points',total=len(images))]
    convex_hull_images, triangle_images, warped_images = triangular_warp(images, landmarks_list, classifier_model.landmark_detector.mean_landmarks, visualize=True)

    for index_id, (convex_hull_image, triangle_image, warped_image) in tqdm(enumerate(zip(convex_hull_images, triangle_images, warped_images)),desc=f'Saving the processed images',total=len(convex_hull_images)):
        image_name = str(df_emotions_test.index[index_id]).zfill(z_fill_length)
        image_output_path = os.path.join(convex_hulls_output_dir, f"{image_name}.png")
        cv2.imwrite(image_output_path, convex_hull_image)
        image_output_path = os.path.join(triangle_images_output_dir, f"{image_name}.png")
        cv2.imwrite(image_output_path, triangle_image)
        image_output_path = os.path.join(normalized_images_output_dir, f"{image_name}.png")
        cv2.imwrite(image_output_path, warped_image)
    
    print("Predicting the emotions.")
    predicted_emotions = classifier_model.predict(images)
    real_emotions = df_emotions_test.values.flatten()

    accuracy = accuracy_score(real_emotions, predicted_emotions)
    f1 = f1_score(real_emotions, predicted_emotions, average="weighted")

    eval_result_output_path = os.path.join(model_output_dir, f"eval_results.txt")
    eval_file = open(eval_result_output_path, "w")
    eval_file.writelines([
        f"Accuracy: {accuracy}\n", 
        f"F1 Score: {f1}\n"])
    eval_file.close()

    labels = sorted(np.unique(np.concatenate((real_emotions, predicted_emotions))))
    cm = confusion_matrix(real_emotions, predicted_emotions,labels=labels)

    df_cm = pd.DataFrame(cm, labels, labels)
    plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
    confusion_matrix_output_path = os.path.join(model_output_dir, f"confusion_matrix.png")
    plt.savefig(confusion_matrix_output_path)

    for index_id, (image, normalized_image, real_emotion, predicted_emotion) in tqdm(enumerate(zip(images, warped_images, real_emotions, predicted_emotions)),desc=f'Predicting the emotions',total=len(images)):
        height, width = image.shape
        image = cv2.copyMakeBorder(image, 0, 50, 0, 0, cv2.BORDER_CONSTANT,value=(0,0,0))
        image = cv2.putText(image, f'Real Emotion: {real_emotion.title()}', (width//2-80, height+29), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

        height, width = normalized_image.shape
        normalized_image = cv2.copyMakeBorder(normalized_image, 0, 50, 0, 0, cv2.BORDER_CONSTANT,value=(0,0,0))
        normalized_image = cv2.putText(normalized_image, f'Predicted Emotion: {predicted_emotion.title()}', (width//2-110, height+29), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        
        output_image = np.concatenate((image, normalized_image),axis=1)
        image_name = str(df_emotions_test.index[index_id]).zfill(z_fill_length)
        image_output_path = os.path.join(emotions_output_dir, f"{image_name}.png")
        cv2.imwrite(image_output_path, output_image)

if __name__ == "__main__":
    main()


        