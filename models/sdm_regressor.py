import cv2
from sklearn.linear_model import LinearRegression
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

class SDMRegressor:
    def __init__(self, num_regressors, num_initial_samples, num_landmark_coordinates, sift_patch_size, pca_explained_variance):
        self.num_regressors = num_regressors
        self.num_initial_samples = num_initial_samples
        # num_landmark_coordinates = 2*num_landmarks (x and y coordinates for each point)
        self.num_landmark_coordinates = num_landmark_coordinates
        self.cascaded_regressors = [[LinearRegression() for _ in range(num_regressors)] for _ in range(self.num_landmark_coordinates)]
        self.pca_list = [PCA(n_components=pca_explained_variance) for _ in range(num_regressors)]
        self.sift_patch_size = sift_patch_size

    # Monte Carlo sampling
    def get_initial_images_landmarks_sets(self, df_landmarks):
        mean = np.asarray(df_landmarks.mean(axis=0))
        self.mean_landmarks = mean.astype(int).reshape(-1,2)
        cov = np.asarray(df_landmarks.cov())
        num_images = len(df_landmarks)
        # initial landmark locations for each coordinate <num_images * num_initial_samples * [x1, y1, x2, y2 ..., xn, yn]> of n landmarks
        initial_images_landmark_sets = np.asarray([np.random.multivariate_normal(mean, cov, self.num_initial_samples).astype(int) for _ in range(num_images)])
        # returned array has three dimensions: (num_images, num_samples, num_landmark_coordinates)
        return initial_images_landmark_sets

    def extract_images_sift_descriptors(self, images, images_landmarks_sets):
        sift = cv2.SIFT_create()
        images_sift_descriptors = []
        for image, landmarks_sets in tqdm(zip(images, images_landmarks_sets),desc=f'Extracting SIFT Descriptors',total=len(images)):
            image_descriptors = []
            for landmarks_set in landmarks_sets:
                # pairing the x and y coordinates to find the descriptors of a point
                landmarks_set_pair = landmarks_set.reshape(-1,2) 
                # converting landmarks into cv library format to be able to extract the sift descriptors
                cv_landmarks = []
                for x,y in landmarks_set_pair:
                    cv_landmark = cv2.KeyPoint(float(x),float(y),self.sift_patch_size) 
                    # for both x and y coordinates, the descriptor will be the same
                    # therefore, appending descriptor (will be calcuated in the next line) two times 
                    # as the input of the regressor for both x and y coordinates
                    cv_landmarks += [cv_landmark, cv_landmark]
                _, sift_descriptors = sift.compute(image, cv_landmarks)
                image_descriptors.append(sift_descriptors)
            images_sift_descriptors.append(image_descriptors)
        images_sift_descriptors = np.asarray(images_sift_descriptors)
        # the objects are deleted to be able to save the trained model by using pickle
        # pickle cannot dump the cv2 objects
        del cv_landmarks
        del sift
        # returned array has four dimensions: (num_images, num_samples, num_landmark_coordinates, descriptor_size)
        return images_sift_descriptors

    def get_target_landmarks_set(self, df_landmarks):
        df_landmarks_array = df_landmarks.values
        images_target_landmarks_sets = []
        for image_target_landmarks in df_landmarks_array:
            image_target_landmarks_sets = []
            # for each initial landmark sets of an image, appending the target values
            for _ in range(self.num_initial_samples):
                image_target_landmarks_sets.append(image_target_landmarks)
            images_target_landmarks_sets.append(image_target_landmarks_sets)
        images_target_landmarks_sets = np.asarray(images_target_landmarks_sets)
        # returned array has three dimensions: (num_images, num_samples, num_landmark_coordinates)
        return images_target_landmarks_sets

    def fit(self, images, df_landmarks):
        images_current_landmarks_sets = self.get_initial_images_landmarks_sets(df_landmarks)
        for regressor_id in range(self.num_regressors):
            print("Extracting the sift descriptors of the current landmark points on each image")
            pca = self.pca_list[regressor_id]
            images_sift_descriptors = self.extract_images_sift_descriptors(images, images_current_landmarks_sets)
            num_images, num_samples, num_landmark_coordinates, descriptor_size = images_sift_descriptors.shape
            descriptors = images_sift_descriptors.reshape(num_images*num_samples,descriptor_size*self.num_landmark_coordinates)
            print("Applying PCA on the SIFT descriptors!")
            pca.fit(descriptors)
            descriptors = pca.transform(descriptors) # dimensionality reduction using pca
            print("PCA is applied on the SIFT descriptors!")
            images_target_landmarks_sets = self.get_target_landmarks_set(df_landmarks)
            images_delta_landmarks_sets = images_target_landmarks_sets - images_current_landmarks_sets
            print(f"Training the regressor {regressor_id+1} using the extracted descriptors")
            for landmark_coordinate_id in tqdm(range(self.num_landmark_coordinates),desc=f'Training Regressor {regressor_id+1}',total=self.num_landmark_coordinates):
                target_delta_landmark_coordinates = images_delta_landmarks_sets[:,:,landmark_coordinate_id].flatten()
                self.cascaded_regressors[landmark_coordinate_id][regressor_id].fit(descriptors, target_delta_landmark_coordinates)
                # predicting the delta values by using the trained regressor
                predicted_delta_landmark_coordinates = self.cascaded_regressors[landmark_coordinate_id][regressor_id].predict(descriptors)
                images_current_landmarks_sets[:,:,landmark_coordinate_id] += predicted_delta_landmark_coordinates.reshape(num_images, num_samples).astype(int)

    def extract_sift_descriptors(self, image, landmarks_set):
        sift = cv2.SIFT_create()
        # pairing the x and y coordinates to find the descriptors of a point
        landmarks_set_pair = landmarks_set.reshape(-1,2) 
        # converting landmarks into cv library format to be able to extract the sift descriptors
        cv_landmarks = []
        for x,y in landmarks_set_pair:
            cv_landmark = cv2.KeyPoint(float(x),float(y), self.sift_patch_size) 
            # for both x and y coordinates, the descriptor will be the same
            # therefore, appending descriptor (will be calcuated in the next line) two times 
            # as the input of the regressor for both x and y coordinates
            cv_landmarks += [cv_landmark, cv_landmark]
        _, sift_descriptors = sift.compute(image, cv_landmarks)
        # the objects are deleted to be able to save the trained model by using pickle
        # pickle cannot dump the cv2 objects
        del cv_landmarks
        del sift
        return sift_descriptors

    def predict(self, image):
        current_landmarks = self.mean_landmarks.flatten()
        for regressor in range(self.num_regressors):
            pca = self.pca_list[regressor]
            descriptors = self.extract_sift_descriptors(image,current_landmarks)
            descriptors = [descriptors.flatten()]
            descriptors = pca.transform(descriptors) # dimensionality reduction using pca
            for landmark_coordinate_id in range(self.num_landmark_coordinates):
                delta_coordinate = self.cascaded_regressors[landmark_coordinate_id][regressor].predict(descriptors)[0]
                current_landmarks[landmark_coordinate_id] += int(delta_coordinate)
        return current_landmarks.reshape(-1,2)