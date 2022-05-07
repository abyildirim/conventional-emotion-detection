# Conventional Emotion Detection
The aim of this project is to detect emotion of a person from a given visual input by using the conventional computer vision methods.

## Emotion Detection Pipeline
1. Facial landmark detection 
    * Regression-based model
    * [Supervised Descent Method](https://www.ri.cmu.edu/pub_files/2013/5/main.pdf)
2. **TBD**

## Team Members
* Ahmet Burak Yıldırım
* Maria Raluca Duminică


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
The downloaded datasets are saved to the ***datasets*** directory.

3. Preparing the [LFPW](https://docs.activeloop.ai/datasets/lfpw-dataset) dataset

    ```
    cd dataset_preprocessors
    python lfpw_preprocessor.py
    ```

## Training
The trained models are saved to the ***saved_models*** directory by default.

4. Training the detection model **(For now, it is only the facial landmark detection model)**

    ```
    cd ..
    python train.py
    ```
    *Arguments:*
    * *--dataset:* Name of the dataset.
    * *--dataset_dir:* Directory of the dataset that the regressor is trained on.
    * *--num_regressors:* Number of cascaded regressors for each landmark coordinate (different regressors used for x and y coordinates of the landmarks)).
    * *--num_initial_samples:* Number of landmark points sets created initially for each image in the training phase.
    * *--sift_patch_size:* Patch size used to extract the sift descriptors of the landmarks.
    * *--model_save_dir:* Where the trained model will be saved.
    * *--pca_explained_variance:* Explained variance of PCA components of the SIFT descriptors used for training the regressor models.

## Evaluating
The evaluation results of the models are saved to the ***output*** directory  by default.

4. Evaluating the trained detection model **(For now, it is only the facial landmark detection model)**

    ```
    python eval.py
    ```
    *Arguments:*
    * *--dataset:* Name of the dataset.
    * *--dataset_dir:* Directory of the dataset that the regressor is trained on.
    * *--output_dir:* Where the results will be saved.
    * *--model_dir:* Where the trained model will be loaded.
    * *--model_name:* Name of the model in the saved models' directory.

