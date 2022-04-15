import hub
import cv2
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

global haar_xml_path
global haar_cascade_classifier
haar_xml_path = os.path.join(os.path.dirname( __file__ ), "..", "haarcascade_frontalface_default.xml")
haar_cascade_classifier = cv2.CascadeClassifier(haar_xml_path)

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

# https://docs.activeloop.ai/datasets/lfpw-dataset
def process_and_save_data(hub_path, data_type):
    print("# Preprocessing operations on the", data_type, "data is started!")
    landmarks_dict = {}
    dataset = hub.load(hub_path)
    dataloader = dataset.pytorch(num_workers=0, batch_size=1, shuffle=False)
    zfill_length = len(str(len(dataset)))
    dataset_dir = os.path.join(os.path.dirname( __file__ ), "..", "datasets", "lfpw")
    landmarks_dir = os.path.join(dataset_dir, "processed", data_type)
    image_processed_dir = os.path.join(dataset_dir, "processed", data_type, "images")
    os.makedirs(image_processed_dir, exist_ok=True)
    image_original_dir = os.path.join(dataset_dir, "original", data_type)
    os.makedirs(image_original_dir, exist_ok=True)
    image_landmarks_dir = os.path.join(dataset_dir, "landmarks", data_type)
    os.makedirs(image_landmarks_dir, exist_ok=True)
    
    image_ids = []
    for image_id, (image, keypoints) in tqdm(enumerate(dataloader),desc='Processing Images and Landmarks',total=len(dataloader)):
        image = image[0].numpy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_name = str(image_id).zfill(zfill_length) + ".png"
        image_original_save_path = os.path.join(image_original_dir, image_name)
        cv2.imwrite(image_original_save_path, image)
        keypoints = keypoints[0].numpy().reshape(-1, 3)[:,:2]
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
            image_ids.append(image_id)
    landmarks_df = pd.DataFrame(landmarks_dict)
    landmarks_df.index = image_ids
    df_name = "landmarks.csv"
    landmarks_save_path = os.path.join(landmarks_dir, df_name)
    landmarks_df.to_csv(landmarks_save_path)
    print("# The data is saved to:", dataset_dir)

def main():
    training_data_hub_path = "hub://activeloop/LFPW-train"
    test_data_hub_path = "hub://activeloop/LFPW-test"
    
    process_and_save_data(training_data_hub_path, "train")
    process_and_save_data(test_data_hub_path, "test")

if __name__ == "__main__":
    main()