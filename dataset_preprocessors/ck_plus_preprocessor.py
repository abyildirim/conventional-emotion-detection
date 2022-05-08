import cv2
import os
import pandas as pd
from sklearn import datasets
from tqdm import tqdm
import numpy as np
from PIL import Image

global haar_xml_path
global haar_cascade_classifier
global emotions_dict

haar_xml_path = os.path.join(os.path.dirname( __file__ ), "..", "haarcascade_frontalface_default.xml")
haar_cascade_classifier = cv2.CascadeClassifier(haar_xml_path)
emotions_dict = {
    0:"neutral", 
    1:"anger", 
    2:"contempt", 
    3:"disgust", 
    4:"fear", 
    5:"happy", 
    6:"sadness",
    7:"surprise"
}

def crop_face(image, keypoints):
    if image.shape[2] != 1:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image
    face_boxes = haar_cascade_classifier.detectMultiScale(image_gray)
    if len(face_boxes) == 0:
        return None, None
    landmark_top_left_corner = keypoints.min(axis=0)
    selected_face_box = None
    min_face_box_distance = np.inf
    for x_left, y_top, width, height in face_boxes:
        face_box = np.asarray([x_left, y_top, width, height])
        top_left_corner = face_box[:2]
        face_box_distance = np.linalg.norm(landmark_top_left_corner-top_left_corner)
        if face_box_distance < min_face_box_distance:
            min_face_box_distance = face_box_distance
            selected_face_box = face_box
    x_left, y_top, width, height = selected_face_box
    num_inlier_landmarks = (np.all(keypoints > selected_face_box[:2], axis=1) & np.all(keypoints < (selected_face_box[:2] + selected_face_box[2:]), axis=1)).sum()
    inlier_ratio_treshold = 0.7
    if num_inlier_landmarks/len(keypoints) < inlier_ratio_treshold:
        return None, None
    cropped_image = image[y_top:y_top+height, x_left:x_left+width, :]
    resized_cropped_image = cv2.resize(cropped_image, (256,256), interpolation = cv2.INTER_NEAREST)
    x_ratio, y_ratio = 256 / cropped_image.shape[0], 256 / cropped_image.shape[1]
    for landmark_id, (x, y) in enumerate(keypoints):
        keypoints[landmark_id] = (x-x_left) * x_ratio, (y-y_top) * y_ratio
    return resized_cropped_image, keypoints

def read_emotion(image_sequence_id_path):
    emotion = None
    emotion_file_list = os.listdir(image_sequence_id_path)
    if len(emotion_file_list) != 0:
        emotion_file = emotion_file_list[0]
        emotion_file_path = os.path.join(image_sequence_id_path, emotion_file)
        emotion_file = open(emotion_file_path, "r")
        emotion_id = int(float(emotion_file.readline().strip()))
        emotion = emotions_dict[emotion_id]
        emotion_file.close()
    return emotion

def read_landmarks(landmarks_folder_path, is_peak):
    file_index = 0
    if is_peak:
        file_index = -1
    landmarks_file_list = os.listdir(landmarks_folder_path)
    landmarks_file_list.sort()
    landmarks_file_path = os.path.join(landmarks_folder_path, landmarks_file_list[file_index])
    landmarks_file = open(landmarks_file_path, "r")
    landmarks = []
    for landmark_line in landmarks_file:
        x, y = landmark_line.strip().split()
        x, y = int(float(x)), int(float(y))
        landmarks.append((x, y))
    return landmarks

def read_peak_image(images_folder_path):
    images_list = os.listdir(images_folder_path)
    images_list.sort()
    peak_image_path = os.path.join(images_folder_path, images_list[-1])
    peak_image = Image.open(peak_image_path).convert("RGB")
    return peak_image

def read_neutral_image(images_folder_path):
    images_list = os.listdir(images_folder_path)
    images_list.sort()
    neutral_image_path = os.path.join(images_folder_path, images_list[0])
    neutral_image = Image.open(neutral_image_path).convert("RGB")
    return neutral_image

def read_data(training_data_path, setup_number):
    print(f"# Reading the CK+ dataset! (Setup {setup_number})")
    emotions_path = os.path.join(training_data_path, "emotions")
    landmarks_path = os.path.join(training_data_path, "landmarks")
    images_path = os.path.join(training_data_path, "images")
    subjects = os.listdir(emotions_path)
    image_list, landmarks_list, emotion_list = [], [], []
    for subject in subjects:
        subject_path = os.path.join(emotions_path, subject)
        image_sequence_ids = os.listdir(subject_path)
        for image_sequence_id in image_sequence_ids:
            image_sequence_id_path = os.path.join(subject_path, image_sequence_id)
            emotion = read_emotion(image_sequence_id_path)
            if emotion is not None:
                landmarks_folder_path = os.path.join(landmarks_path, subject, image_sequence_id)
                images_folder_path = os.path.join(images_path, subject, image_sequence_id) 
                if setup_number == 2:
                    neutral_image = read_neutral_image(images_folder_path)
                    image_list.append(neutral_image)
                    emotion_list.append(emotions_dict[0])
                    landmarks = read_landmarks(landmarks_folder_path, is_peak=False)
                    landmarks_list.append(landmarks)
                if setup_number == 2 and emotion == emotions_dict[2]:
                    continue
                peak_image = read_peak_image(images_folder_path)
                image_list.append(peak_image)
                emotion_list.append(emotion)
                landmarks = read_landmarks(landmarks_folder_path, is_peak=True)
                landmarks_list.append(landmarks)

    return image_list, landmarks_list, emotion_list

def split_data(image_list, landmarks_list, emotion_list, test_ratio=0.2):
    image_list, landmarks_list, emotion_list = np.asarray(image_list), np.asarray(landmarks_list), np.asarray(emotion_list)
    shuffle_ids = np.arange(len(image_list))
    np.random.shuffle(shuffle_ids)
    image_list, landmarks_list, emotion_list = image_list[shuffle_ids], landmarks_list[shuffle_ids], emotion_list[shuffle_ids]
    dataset_length = len(image_list)
    test_start_index  = int(dataset_length * (1-test_ratio))
    train_dataset  = {
        "images": image_list[:test_start_index],
        "landmarks": landmarks_list[:test_start_index],
        "emotions": emotion_list[:test_start_index],
    }
    test_dataset  = {
        "images": image_list[test_start_index:],
        "landmarks": landmarks_list[test_start_index:],
        "emotions": emotion_list[test_start_index:],
    }
    return train_dataset, test_dataset

def process_and_save_data(dataset, data_type, setup_number):
    print("# Preprocessing operations on the", data_type, "data is started!")
    image_list, landmarks_list, emotion_list = dataset["images"], dataset["landmarks"], dataset["emotions"]
    zfill_length = len(str(len(dataset)))
    dataset_dir = os.path.join(os.path.dirname( __file__ ), "..", "datasets", "ck_plus", f"setup_{setup_number}")
    processed_data_dir = os.path.join(dataset_dir, "processed", data_type)
    image_processed_dir = os.path.join(processed_data_dir, "images")
    os.makedirs(image_processed_dir, exist_ok=True)
    image_original_dir = os.path.join(dataset_dir, "original", data_type)
    os.makedirs(image_original_dir, exist_ok=True)
    image_landmarks_dir = os.path.join(dataset_dir, "landmarks", data_type)
    os.makedirs(image_landmarks_dir, exist_ok=True)

    landmarks_dict = {}
    emotions_list = []
    image_ids = []
    for image_id, (image, keypoints, emotion) in tqdm(enumerate(zip(image_list, landmarks_list, emotion_list)),desc='Processing Images and Landmarks',total=len(image_list)):
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_name = str(image_id).zfill(zfill_length) + ".png"
        image_original_save_path = os.path.join(image_original_dir, image_name)
        cv2.imwrite(image_original_save_path, image)
        keypoints = np.asarray(keypoints).reshape(-1, 2)
        image, keypoints = crop_face(image, keypoints)
        if image is not None:
            image_processed_save_path = os.path.join(image_processed_dir, image_name)
            cv2.imwrite(image_processed_save_path, image)
            for landmark_id, (x, y) in enumerate(keypoints):
                x_column = "x_{}".format(landmark_id)
                y_column = "y_{}".format(landmark_id)
                try:
                    landmarks_dict[x_column].append(x)
                    landmarks_dict[y_column].append(y)
                except:
                    landmarks_dict[x_column] = [x]
                    landmarks_dict[y_column] = [y]
                color = (0, 255, 0)
                cv2.circle(image, (x, y), 1, color, 2)
            image_landmarks_save_path = os.path.join(image_landmarks_dir, image_name)
            cv2.imwrite(image_landmarks_save_path, image)
            emotions_list.append(emotion)
            image_ids.append(image_id)

    landmarks_df = pd.DataFrame(landmarks_dict)
    landmarks_df.index = image_ids
    df_name = "landmarks.csv"
    landmarks_save_path = os.path.join(processed_data_dir, df_name)
    landmarks_df.to_csv(landmarks_save_path)

    emotions_df = pd.DataFrame({"emotion": emotion_list})
    emotions_df.index = image_ids
    df_name = "emotions.csv"
    emotions_save_path = os.path.join(processed_data_dir, df_name)
    emotions_df.to_csv(emotions_save_path)
    print("# The data is saved to:", dataset_dir)

def main():
    root_path = os.path.join(os.path.dirname( __file__ ), "..")
    training_data_path = os.path.join(root_path, "datasets", "ck_plus", "raw")
    image_list, landmarks_list, emotion_list = read_data(training_data_path, setup_number=1)
    train_dataset, test_dataset = split_data(image_list, landmarks_list, emotion_list)
    process_and_save_data(train_dataset, "train", setup_number=1)
    process_and_save_data(test_dataset, "test", setup_number=1)

    image_list, landmarks_list, emotion_list = read_data(training_data_path, setup_number=2)
    train_dataset, test_dataset = split_data(image_list, landmarks_list, emotion_list)
    process_and_save_data(train_dataset, "train", setup_number=2)
    process_and_save_data(test_dataset, "test", setup_number=2)


if __name__ == "__main__":
    main()