import cv2
import numpy as np
import pickle
import os
import argparse

global haar_xml_path
global haar_cascade_classifier
haar_xml_path = os.path.join(os.path.dirname( __file__ ), "haarcascade_frontalface_default.xml")
haar_cascade_classifier = cv2.CascadeClassifier(haar_xml_path)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_id', type=int, default=0, help='Camera input id.')
    parser.add_argument('--dataset', type=str, default="ck_plus_setup_2", help='Name of the dataset.')
    parser.add_argument('--model_dir', type=str, default="./saved_models", help='Where the trained models will be loaded.')
    parser.add_argument('--model_name', type=str, default="emotion_detector_logistic_regressor_pca0.5", help='Name of the emotion detector model in the saved models\' directory.')
    args = parser.parse_args()

    # python realtime_test.py --dataset ck_plus_setup_1 --model_dir ./saved_models --model_name emotion_detector_svm_pca0.98
    # python realtime_test.py --dataset ck_plus_setup_1 --model_dir ./saved_models --model_name emotion_detector_decision_tree_pca0.98
    # python realtime_test.py --dataset ck_plus_setup_1 --model_dir ./saved_models --model_name emotion_detector_knn_pca0.98
    # python realtime_test.py --dataset ck_plus_setup_1 --model_dir ./saved_models --model_name emotion_detector_logistic_regressor_pca0.98
    # python realtime_test.py --dataset ck_plus_setup_1 --model_dir ./saved_models --model_name emotion_detector_svm_pca0.5
    # python realtime_test.py --dataset ck_plus_setup_1 --model_dir ./saved_models --model_name emotion_detector_decision_tree_pca0.5
    # python realtime_test.py --dataset ck_plus_setup_1 --model_dir ./saved_models --model_name emotion_detector_knn_pca0.5
    # python realtime_test.py --dataset ck_plus_setup_1 --model_dir ./saved_models --model_name emotion_detector_logistic_regressor_pca0.5

    # python realtime_test.py --dataset ck_plus_setup_2 --model_dir ./saved_models --model_name emotion_detector_svm_pca0.98
    # python realtime_test.py --dataset ck_plus_setup_2 --model_dir ./saved_models --model_name emotion_detector_decision_tree_pca0.98
    # python realtime_test.py --dataset ck_plus_setup_2 --model_dir ./saved_models --model_name emotion_detector_knn_pca0.98
    # python realtime_test.py --dataset ck_plus_setup_2 --model_dir ./saved_models --model_name emotion_detector_logistic_regressor_pca0.98
    # python realtime_test.py --dataset ck_plus_setup_2 --model_dir ./saved_models --model_name emotion_detector_svm_pca0.5
    # python realtime_test.py --dataset ck_plus_setup_2 --model_dir ./saved_models --model_name emotion_detector_decision_tree_pca0.5
    # python realtime_test.py --dataset ck_plus_setup_2 --model_dir ./saved_models --model_name emotion_detector_knn_pca0.5
    # python realtime_test.py --dataset ck_plus_setup_2 --model_dir ./saved_models --model_name emotion_detector_logistic_regressor_pca0.5

    checkpoint_dir = os.path.join(args.model_dir, args.dataset, f"{args.model_name}.model")

    with open(checkpoint_dir, 'rb') as model_file:
        model = pickle.load(model_file)
        
    cap = cv2.VideoCapture(args.camera_id)
    width, height = 1280, 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    _, image = cap.read()
    assert image.shape[0] == height
    assert image.shape[1] == width
    cv2.imshow('Emotion Detector', image)

    while cap.isOpened():
        # Close the application when the window is closed
        try:
            cv2.getWindowProperty('Emotion Detector', 0)
        except:
            break
        
        # Read frame from the camera input
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        if image.shape[2] != 1:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image
        face_boxes = haar_cascade_classifier.detectMultiScale(image_gray)

        # If face is not found, read the next frame
        if len(face_boxes) == 0:
            continue
        
        # Take the first detected image (assuming there is only one face)
        face_box = face_boxes[0]
        x_left, y_top, width, height = face_box
        face_box = np.asarray([x_left, y_top, width, height])
        cropped_image = image_gray[y_top:y_top+height, x_left:x_left+width]
        resized_cropped_image = cv2.resize(cropped_image, (256,256), interpolation = cv2.INTER_NEAREST)
        image = resized_cropped_image

        predicted_landmarks, normalized_images, predicted_emotions = model.predict([image], return_visuals=True)
        predicted_landmarks, normalized_image, predicted_emotion = predicted_landmarks[0], normalized_images[0], predicted_emotions[0]

        landmark_image = image.copy()
        for x,y in predicted_landmarks:
            cv2.circle(landmark_image, (x,y), 1, (0, 255, 0), 2)
        
        landmark_image = cv2.flip(landmark_image, 1)
        image = cv2.flip(image, 1)
        normalized_image = cv2.flip(normalized_image, 1)
        image = np.concatenate((landmark_image, image, normalized_image), axis=1)
        height, width = image.shape
        image = cv2.copyMakeBorder(image, 0, 50, 0, 0, cv2.BORDER_CONSTANT,value=(0,0,0))
        image = cv2.putText(image, f'Predicted Emotion: {predicted_emotion.title()}', (width//2-110, height+29), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow('Emotion Detector', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()

if __name__ == "__main__":
    main()