# Conventional Emotion Detection
The aim of this project is to detect the emotion of a person from a given visual input by using conventional computer vision methods.

## Team Members
* Ahmet Burak Yıldırım
* Maria Raluca Duminică

## Emotion Detection Pipeline
1. Facial Feature Point Detection
    * Preprocessing: Viola-Jones face detection + resizing (256x256)
    * [Regression-based Supervised Descent Method (SDM)](https://www.ri.cmu.edu/pub_files/2013/5/main.pdf)

    <br />

    ![SDM](/images/sdm.png)

2. Appearance Processing
    * Image normalization: Triangular warping
    * HOG descriptors’ extraction

    <br />

    ![Appearance Model](/images/appearance_model.png)

3. Emotion Detection
    * Features: HOG descriptors + landmarks
    * Classifiers
        * Logistic Regression
        * SVM
        * Decision Tree
        * KNN

    <br />

    ![Emotion Detection](/images/emotion_detection.png)

# Installation

Python 3.8 is used in the experiments.

## Environment Setup

1. Cloning the GitHub repository

    ```
    git clone https://github.com/abyildirim/conventional-emotion-detection.git
    ```

2. Installing the dependencies

    ```
    cd conventional-emotion-detection
    pip install -r requirements.txt
    ```
## Dataset Preprocessing

3. Preparing the [LFPW](https://docs.activeloop.ai/datasets/lfpw-dataset) dataset

    ```
    cd dataset_preprocessors
    python lfpw_preprocessor.py
    ```
    * The dataset is saved to the `<project_root>/datasets/lfpw/` directory.

4. CK+ dataset is not publicly available. It can be requested from [this link](https://sites.pitt.edu/~emotion/ck-spread.htm). In order to apply the preprocessing operations:

    ```
    cd dataset_preprocessors
    python ck_plus_preprocessor.py
    ```
    * Note that the dataset should be extracted to the `<project_root>/datasets/ck_plus/raw/` directory before running the script.
    * There are two setups used to be able to compare the results with the other studies. Images of five main emotions exist in both setups, which are anger, disgust, fear, happiness, sadness, and surprise. Additionally, Setup 1 contains images of the contempt emotion while Setup 2 contains images of the neutral emotion.

Visualization of the preprocessing operations on both datasets:
* Viola-Jones face detection
* Resizing the image (256x256)
* Resizing the facial feature points (256x256)

![Dataset Preprocessing](/images/dataset_preprocessing.png)

## Training & Evaluating

Example commands are provided in the following sections to run the scripts. Commands used to train and evaluate each model are written as comments inside these scripts.

### Training
Trained models are saved to the `<project_root>/saved_models/` directory by default.

5. Training a facial landmark detection model

    ```
    python train_landmark_detector.py --dataset ck_plus_setup_2 --dataset_dir ./datasets/ck_plus/setup_2/processed --num_regressors 5 --num_initial_samples 10 --sift_patch_size 32 --model_save_dir ./saved_models --pca_explained_variance 0.98
    ```

6. Training an emotion detection model

    ```
    python train_emotion_detector.py --dataset ck_plus_setup_2 --classifier logistic_regressor --dataset_dir ./datasets/ck_plus/setup_2/processed --model_save_dir ./saved_models --pca_explained_variance 0.98
    ```

### Evaluating
The evaluation results of the models are saved to the `<project_root>/output/` directory by default.

7. Evaluating a facial landmark detection model

    ```
    python evaluate_landmark_detector.py --dataset ck_plus_setup_2 --dataset_dir ./datasets/ck_plus/setup_2/processed --output_dir ./output --model_dir ./saved_models --model_name sdm_landmark_regressor_r5_p32_s10_pca0.98
    ```

8. Evaluating an emotion detection model

    ```
    python evaluate_emotion_detector.py --dataset ck_plus_setup_2 --model_name emotion_detector_logistic_regressor_pca0.98 --dataset_dir ./datasets/ck_plus/setup_2/processed --output_dir ./output --model_dir ./saved_models
    ```

## Example Results

### Facial Landmark Detection

![SDM Results](/images/sdm_results.png)

### Emotion Detection (CK+)

![Emotion Detection Results](/images/emotion_detection_results.png)

## Running Realtime Test Using Webcam

You should be looking directly at the camera. The lightning conditions and the camera resolution may affect the performance of the model.

```
python realtime_test.py --dataset ck_plus_setup_2 --model_dir ./saved_models --model_name emotion_detector_logistic_regressor_pca0.98

```
* Only the logistic regression model trained on the CK+ Setup 2 dataset is provided in the repository. You can train the models using different configurations as explained above sections.