from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog
from tqdm import tqdm
from sklearn.decomposition import PCA
from utils.triangular_warper import triangular_warp
from sklearn.preprocessing import StandardScaler
import numpy as np

class EmotionDetector:
    def __init__(self, classifier_type, landmark_detector, pca_explained_variance):
        if classifier_type == "svm":
            self.classifier = SVC(decision_function_shape='ovo', C=0.8, kernel="sigmoid")
        elif classifier_type == "random_forest":
            self.classifier = RandomForestClassifier(n_estimators=10, max_depth=20, random_state=0)
        elif classifier_type == "knn":
            self.classifier = KNeighborsClassifier(n_neighbors=10, weights="distance")
        elif classifier_type == "logistic_regressor":
            self.classifier = LogisticRegression(multi_class='multinomial', random_state=0, C=0.8, class_weight="balanced")
        else:
            raise Exception("{} is not a valid classifier type!".format(classifier_type))
        self.pca = PCA(n_components=pca_explained_variance)
        self.landmark_detector = landmark_detector
        self.emotion_id_dict = {
            "neutral":0, 
            "anger":1, 
            "contempt":2, 
            "disgust":3, 
            "fear":4, 
            "happy":5, 
            "sadness":6,
            "surprise":7
        }
        self.landmark_scaler = StandardScaler()
        self.hog_scaler = StandardScaler()

    def fit(self, images, df_emotions):
        target_emotions = df_emotions["emotion"].tolist()
        target_emotions = [self.emotion_id_dict[emotion] for emotion in target_emotions]
        landmarks = [self.landmark_detector.predict(image) for image in tqdm(images,desc='Predicting landmark points',total=len(images))]
        normalized_images = triangular_warp(images, landmarks, self.landmark_detector.mean_landmarks)
        hog_descriptors = [hog(image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), multichannel=False) for image in normalized_images]
        print("Applying PCA on the HOG descriptors!")
        self.pca.fit(hog_descriptors)
        hog_descriptors = self.pca.transform(hog_descriptors) # dimensionality reduction using pca
        print("PCA is applied on the HOG descriptors!")
        flattened_landmarks = np.asarray(landmarks).reshape(len(landmarks),-1)
        self.landmark_scaler.fit(flattened_landmarks)
        scaled_landmarks = self.landmark_scaler.transform(flattened_landmarks)
        self.hog_scaler.fit(hog_descriptors)
        scaled_hog_descriptors = self.hog_scaler.transform(hog_descriptors)
        hog_landmark_concat = [np.concatenate((hog,landmark)) for hog,landmark in zip(scaled_hog_descriptors, scaled_landmarks)]
        self.classifier.fit(hog_landmark_concat, target_emotions)

    def predict(self, images, return_visuals=False):
        predicted_landmarks = [self.landmark_detector.predict(image) for image in tqdm(images,desc='Extracting the landmark points',total=len(images))]
        normalized_images = triangular_warp(images, predicted_landmarks, self.landmark_detector.mean_landmarks)
        hog_descriptors = [hog(normalized_image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), multichannel=False) for normalized_image in normalized_images]
        hog_descriptors = self.pca.transform(hog_descriptors) # dimensionality reduction using pca
        flattened_landmarks = np.asarray(predicted_landmarks).reshape(len(predicted_landmarks),-1)
        scaled_landmarks = self.landmark_scaler.transform(flattened_landmarks)
        scaled_hog_descriptors = self.hog_scaler.transform(hog_descriptors)
        hog_landmark_concat = [np.concatenate((hog,landmark)) for hog,landmark in zip(scaled_hog_descriptors, scaled_landmarks)]
        predicted_emotion_ids = self.classifier.predict(hog_landmark_concat)
        inverse_emotion_id_dict = {v: k for k, v in self.emotion_id_dict.items()}
        predicted_emotions = [inverse_emotion_id_dict[predicted_emotion_id] for predicted_emotion_id in predicted_emotion_ids]
        if return_visuals:
            return predicted_landmarks, normalized_images, predicted_emotions
        return predicted_emotions