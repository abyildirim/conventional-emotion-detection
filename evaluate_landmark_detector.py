from PIL import Image
import numpy as np
import pandas as pd
import os
import pandas as pd
import cv2
import pickle
import argparse
from tqdm import tqdm

def get_images(images_dir):
    print("Reading the images inside the dataset.")
    images = []
    for image_file in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_file)
        image = Image.open(image_path)
        images.append(image)
    images = np.stack(images)
    return images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="lfpw", help='Name of the dataset.')
    parser.add_argument('--dataset_dir', type=str, default="./datasets/lfpw/processed", help='Directory of the dataset that the regressor is trained on.')
    parser.add_argument('--output_dir', type=str, default="./output", help='Where the results will be saved.')
    parser.add_argument('--model_dir', type=str, default="./saved_models", help='Where the trained model will be loaded.')
    parser.add_argument('--model_name', type=str, default="sdm_landmark_regressor_r5_p32_s10_pca0.98", help='Name of the model in the saved models\' directory.')
    args = parser.parse_args()

    # python evaluate_landmark_detector.py --dataset lfpw --dataset_dir ./datasets/lfpw/processed --output_dir ./output --model_dir ./saved_models --model_name sdm_landmark_regressor_r1_p32_s1_pca0.98
    # python evaluate_landmark_detector.py --dataset lfpw --dataset_dir ./datasets/lfpw/processed --output_dir ./output --model_dir ./saved_models --model_name sdm_landmark_regressor_r1_p32_s10_pca0.98
    # python evaluate_landmark_detector.py --dataset lfpw --dataset_dir ./datasets/lfpw/processed --output_dir ./output --model_dir ./saved_models --model_name sdm_landmark_regressor_r5_p32_s1_pca0.98
    # python evaluate_landmark_detector.py --dataset lfpw --dataset_dir ./datasets/lfpw/processed --output_dir ./output --model_dir ./saved_models --model_name sdm_landmark_regressor_r5_p32_s10_pca0.98
    # python evaluate_landmark_detector.py --dataset lfpw --dataset_dir ./datasets/lfpw/processed --output_dir ./output --model_dir ./saved_models --model_name sdm_landmark_regressor_r5_p32_s10_pca0.9
    # python evaluate_landmark_detector.py --dataset lfpw --dataset_dir ./datasets/lfpw/processed --output_dir ./output --model_dir ./saved_models --model_name sdm_landmark_regressor_r10_p32_s1_pca0.98
    # python evaluate_landmark_detector.py --dataset lfpw --dataset_dir ./datasets/lfpw/processed --output_dir ./output --model_dir ./saved_models --model_name sdm_landmark_regressor_r10_p32_s10_pca0.98

    # python evaluate_landmark_detector.py --dataset ck_plus_setup_1 --dataset_dir ./datasets/ck_plus/setup_1/processed --output_dir ./output --model_dir ./saved_models --model_name sdm_landmark_regressor_r1_p32_s1_pca0.98
    # python evaluate_landmark_detector.py --dataset ck_plus_setup_1 --dataset_dir ./datasets/ck_plus/setup_1/processed --output_dir ./output --model_dir ./saved_models --model_name sdm_landmark_regressor_r1_p32_s10_pca0.98
    # python evaluate_landmark_detector.py --dataset ck_plus_setup_1 --dataset_dir ./datasets/ck_plus/setup_1/processed --output_dir ./output --model_dir ./saved_models --model_name sdm_landmark_regressor_r5_p32_s1_pca0.98
    # python evaluate_landmark_detector.py --dataset ck_plus_setup_1 --dataset_dir ./datasets/ck_plus/setup_1/processed --output_dir ./output --model_dir ./saved_models --model_name sdm_landmark_regressor_r5_p32_s10_pca0.98
    # python evaluate_landmark_detector.py --dataset ck_plus_setup_1 --dataset_dir ./datasets/ck_plus/setup_1/processed --output_dir ./output --model_dir ./saved_models --model_name sdm_landmark_regressor_r10_p32_s1_pca0.98
    # python evaluate_landmark_detector.py --dataset ck_plus_setup_1 --dataset_dir ./datasets/ck_plus/setup_1/processed --output_dir ./output --model_dir ./saved_models --model_name sdm_landmark_regressor_r10_p32_s10_pca0.98

    # python evaluate_landmark_detector.py --dataset ck_plus_setup_2 --dataset_dir ./datasets/ck_plus/setup_2/processed --output_dir ./output --model_dir ./saved_models --model_name sdm_landmark_regressor_r1_p32_s1_pca0.98
    # python evaluate_landmark_detector.py --dataset ck_plus_setup_2 --dataset_dir ./datasets/ck_plus/setup_2/processed --output_dir ./output --model_dir ./saved_models --model_name sdm_landmark_regressor_r1_p32_s10_pca0.98
    # python evaluate_landmark_detector.py --dataset ck_plus_setup_2 --dataset_dir ./datasets/ck_plus/setup_2/processed --output_dir ./output --model_dir ./saved_models --model_name sdm_landmark_regressor_r5_p32_s1_pca0.98
    # python evaluate_landmark_detector.py --dataset ck_plus_setup_2 --dataset_dir ./datasets/ck_plus/setup_2/processed --output_dir ./output --model_dir ./saved_models --model_name sdm_landmark_regressor_r5_p32_s10_pca0.98
    # python evaluate_landmark_detector.py --dataset ck_plus_setup_2 --dataset_dir ./datasets/ck_plus/setup_2/processed --output_dir ./output --model_dir ./saved_models --model_name sdm_landmark_regressor_r10_p32_s1_pca0.98
    # python evaluate_landmark_detector.py --dataset ck_plus_setup_2 --dataset_dir ./datasets/ck_plus/setup_2/processed --output_dir ./output --model_dir ./saved_models --model_name sdm_landmark_regressor_r10_p32_s10_pca0.98

    model_output_dir = os.path.join(args.output_dir, args.dataset, args.model_name)
    os.makedirs(model_output_dir,exist_ok=True)
    model_load_path= os.path.join(args.model_dir, args.dataset, f"{args.model_name}.model")
    test_images_dir = os.path.join(args.dataset_dir, "test", "images")
    test_landmarks_path = os.path.join(args.dataset_dir, "test", "landmarks.csv")
    images = get_images(test_images_dir)
    df_landmarks_test = pd.read_csv(test_landmarks_path, index_col=0)

    print("Loading the regressor model.")
    with open(model_load_path, 'rb') as model_file:
        regressor_model = pickle.load(model_file)

    print("Saving the predicted image landmarks and the evaluation result.")
    z_fill_length = len(str(len(images)))
    pred_output_dir = os.path.join(model_output_dir, "pred_results")
    os.makedirs(pred_output_dir,exist_ok=True)
    rmse_list = []
    normalized_rmse_list = [] # using inter-ocular distance
    for index_id, (image, real_landmark_coordinates) in tqdm(enumerate(zip(images,df_landmarks_test.values)),desc=f'Predicting the landmarks',total=len(images)):
        real_landmark_coordinates = real_landmark_coordinates.reshape(-1,2)
        predicted_landmark_coordinates = regressor_model.predict(image)
        mean_landmark_coordinates = regressor_model.mean_landmarks
        mean_image = image.copy()
        pred_image = image.copy()
        real_image = image.copy()
        left_eye_center = (real_landmark_coordinates[41]+real_landmark_coordinates[40]) / 2
        right_eye_center = (real_landmark_coordinates[47]+real_landmark_coordinates[46]) / 2
        inter_ocular_distance = np.linalg.norm(left_eye_center-right_eye_center)
        pred_distances = []
        for (x_mean,y_mean), (x_pred,y_pred), (x_real,y_real) in zip(mean_landmark_coordinates, predicted_landmark_coordinates, real_landmark_coordinates):
            pred_distances.append(np.linalg.norm(np.asarray([x_pred,y_pred])-np.asarray([x_real,y_real])))
            cv2.circle(mean_image, (x_mean,y_mean), 1, (255, 0, 0), 2) # Red
            cv2.circle(real_image, (x_real,y_real), 1, (0, 255, 0), 2) # Green
            cv2.circle(pred_image, (x_pred,y_pred), 1, (0, 0, 255), 2) # Blue
        output_image = np.concatenate((mean_image, pred_image, real_image),axis=1)
        image_name = str(df_landmarks_test.index[index_id]).zfill(z_fill_length)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        image_output_path = os.path.join(pred_output_dir, f"{image_name}.png")
        rmse = np.sqrt(np.mean(np.square(np.asarray(pred_distances))))
        normalized_rmse = rmse/inter_ocular_distance
        height, width, _ = output_image.shape
        output_image = cv2.copyMakeBorder(output_image, 0, 50, 0, 0, cv2.BORDER_CONSTANT,value=(0,0,0))
        output_image = cv2.putText(output_image, f'Normalized RMSE: {round(normalized_rmse,3)}', (width//2-120, height+29), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        cv2.imwrite(image_output_path, output_image)
        rmse_list.append(rmse)
        normalized_rmse_list.append(normalized_rmse)
    rmse_list = np.asarray(rmse_list)
    normalized_rmse_list = np.asarray(normalized_rmse_list)

    mean_rmse = rmse_list.mean()
    max_rmse_im_id = rmse_list.argmax()
    min_rmse_im_id = rmse_list.argmin()
    max_rmse = rmse_list[max_rmse_im_id]
    min_rmse = rmse_list[min_rmse_im_id]
    min_rmse_image_name = df_landmarks_test.index[min_rmse_im_id]
    max_rmse_image_name = df_landmarks_test.index[max_rmse_im_id]

    mean_normalized_rmse = normalized_rmse_list.mean()
    max_normalized_rmse_im_id = normalized_rmse_list.argmax()
    min_normalized_rmse_im_id = normalized_rmse_list.argmin()
    max_normalized_rmse = normalized_rmse_list[max_normalized_rmse_im_id]
    min_normalized_rmse = normalized_rmse_list[min_normalized_rmse_im_id]
    min_normalized_rmse_image_name = df_landmarks_test.index[min_normalized_rmse_im_id]
    max_normalized_rmse_image_name = df_landmarks_test.index[max_normalized_rmse_im_id]

    eval_result_output_path = os.path.join(model_output_dir, f"eval_results.txt")
    eval_file = open(eval_result_output_path, "w")
    eval_file.writelines([
        f"Mean RMSE: {mean_rmse}\n", 
        f"Min RMSE (Image {min_rmse_image_name}): {min_rmse}\n", 
        f"Max RMSE (Image {max_rmse_image_name}): {max_rmse}\n",
        f"Mean Normalized RMSE: {mean_normalized_rmse}\n", 
        f"Min Normalized RMSE (Image {min_normalized_rmse_image_name}): {min_normalized_rmse}\n", 
        f"Max Normalized RMSE (Image {max_normalized_rmse_image_name}): {max_normalized_rmse}\n"])
    eval_file.close()
    
    print("Evaluation is completed!")


if __name__ == "__main__":
    main()


        