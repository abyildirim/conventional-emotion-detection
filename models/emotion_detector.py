from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog
from tqdm import tqdm
from sklearn.decomposition import PCA
from utils.triangular_warper import triangular_warp

class EmotionDetector:
    def __init__(self, classifier_type, landmark_detector, pca_explained_variance):
        if classifier_type == "svm":
            self.classifier = svm.SVC(decision_function_shape='ovo')
        elif classifier_type == "random_forest":
            self.classifier = RandomForestClassifier(random_state=0)
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
        self.classifier.fit(hog_descriptors, target_emotions)

    def predict(self, images):
        predicted_landmarks = [self.landmark_detector.predict(image) for image in tqdm(images,desc='Extracting the landmark points',total=len(images))]
        normalized_images = triangular_warp(images, predicted_landmarks, self.landmark_detector.mean_landmarks)
        hog_descriptors = [hog(normalized_image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), multichannel=False) for normalized_image in normalized_images]
        hog_descriptors = self.pca.transform(hog_descriptors) # dimensionality reduction using pca
        predicted_emotion_ids = self.classifier.predict(hog_descriptors)
        inverse_emotion_id_dict = {v: k for k, v in self.emotion_id_dict.items()}
        predicted_emotions = [inverse_emotion_id_dict[predicted_emotion_id] for predicted_emotion_id in predicted_emotion_ids]
        return predicted_emotions