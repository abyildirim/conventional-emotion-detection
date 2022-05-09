from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

# Adapted from: https://pysource.com/2019/05/09/select-and-warp-triangles-face-swapping-opencv-with-python-part-4/

def triangular_warp(images, landmarks_list, mean_landmarks, visualize=False):
    convex_hull_images = []
    triangle_images = []
    warped_images = []
    for image, landmarks in tqdm(zip(images, landmarks_list),desc='Triangular warp operation is in progress',total=len(images)):
        inlier_landmarks, inlier_mean_landmarks = get_landmarks_in_image_boundary(landmarks, mean_landmarks)
        convex_hull, triangles = get_convexhull_and_triangles(inlier_landmarks)
        triangle_indexes = get_triangle_indexes(inlier_landmarks, triangles)
        warped_image, triangle_image = warp_image(image, inlier_landmarks, inlier_mean_landmarks, triangle_indexes, visualize)
        warped_images.append(warped_image)
        if visualize:
            convex_hull_image = get_convex_hull_image(image, convex_hull)
            convex_hull_images.append(convex_hull_image)
            triangle_images.append(triangle_image)
    if visualize:
        return convex_hull_images, triangle_images, warped_images
    return warped_images

def get_landmarks_in_image_boundary(landmarks, mean_landmarks):
    landmarks_in_boundary = []
    corresponding_mean_landmarks = []
    for (x,y), (x_m, y_m) in zip(landmarks, mean_landmarks):
        if (0 <= x <= 255) and (0 <= y <= 255) and (0 <= x_m <= 255) and (0 <= y_m <= 255):
            landmarks_in_boundary.append((x,y))
            corresponding_mean_landmarks.append((x_m, y_m))
    return np.asarray(landmarks_in_boundary), np.asarray(corresponding_mean_landmarks)

def get_convexhull_and_triangles(landmarks):
    convex_hull = cv2.convexHull(landmarks)
    rectangle = cv2.boundingRect(convex_hull)
    subdiv = cv2.Subdiv2D(rectangle)
    landmarks = landmarks.astype(np.float32)
    subdiv.insert(landmarks)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    return convex_hull, triangles

def get_convex_hull_image(image, convex_hull):
    image = image.copy()
    mask = np.zeros_like(image)
    cv2.fillConvexPoly(mask, convex_hull, 255)
    convex_hull_image = cv2.bitwise_and(image, image, mask=mask)
    return convex_hull_image

def get_triangle_indexes(landmarks, triangles):
    triangle_indexes = []
    for t in triangles:
        point_1 = (t[0], t[1])
        point_2 = (t[2], t[3])
        point_3 = (t[4], t[5])
        point_1_index = np.where((landmarks == point_1).all(axis=1))[0][0]
        point_2_index = np.where((landmarks == point_2).all(axis=1))[0][0]
        point_3_index = np.where((landmarks == point_3).all(axis=1))[0][0]
        triangle = [point_1_index, point_2_index, point_3_index]
        triangle_indexes.append(triangle)
    return triangle_indexes

def warp_image(image, landmarks, mean_landmarks, triangle_indexes, visualize=False):
    warped_image = np.zeros_like(image)
    triangle_image = image.copy()
    warped_image_mask = np.zeros_like(image)+255

    for triangle_index in triangle_indexes:
        # Triangulation of the landmarks
        landmark_triangle_point_1 = landmarks[triangle_index[0]]
        landmark_triangle_point_2 = landmarks[triangle_index[1]]
        landmark_triangle_point_3 = landmarks[triangle_index[2]]
        landmark_triangle = np.array([landmark_triangle_point_1, landmark_triangle_point_2, landmark_triangle_point_3], np.int32)
        landmark_rectangle = cv2.boundingRect(landmark_triangle)
        (x, y, w, h) = landmark_rectangle
        cropped_triangle = image[y: y + h, x: x + w]
        cropped_landmark_triangle_mask = np.zeros((h, w), np.uint8)
        landmark_triangle_points = np.array([[landmark_triangle_point_1[0] - x, landmark_triangle_point_1[1] - y],
                                             [landmark_triangle_point_2[0] - x, landmark_triangle_point_2[1] - y],
                                             [landmark_triangle_point_3[0] - x, landmark_triangle_point_3[1] - y]], np.int32)
        cv2.fillConvexPoly(cropped_landmark_triangle_mask, landmark_triangle_points, 255)

        # Triangulation of the mean landmarks
        mean_landmark_triangle_point_1 = mean_landmarks[triangle_index[0]]
        mean_landmark_triangle_point_2 = mean_landmarks[triangle_index[1]]
        mean_landmark_triangle_point_3 = mean_landmarks[triangle_index[2]]
        mean_landmark_triangle = np.array([mean_landmark_triangle_point_1, mean_landmark_triangle_point_2, mean_landmark_triangle_point_3], np.int32)
        mean_landmark_rectangle = cv2.boundingRect(mean_landmark_triangle)
        (x, y, w, h) = mean_landmark_rectangle
        cropped_mean_landmark_triangle_mask = np.zeros((h, w), np.uint8)
        mean_landmark_triangle_points = np.array([[mean_landmark_triangle_point_1[0] - x, mean_landmark_triangle_point_1[1] - y],
                                                  [mean_landmark_triangle_point_2[0] - x, mean_landmark_triangle_point_2[1] - y],
                                                  [mean_landmark_triangle_point_3[0] - x, mean_landmark_triangle_point_3[1] - y]], np.int32)
        cv2.fillConvexPoly(cropped_mean_landmark_triangle_mask, mean_landmark_triangle_points, 255)

        # Warping landmark triangle to mean landmark triangle
        landmark_triangle_points = np.float32(landmark_triangle_points)
        mean_landmark_triangle_points = np.float32(mean_landmark_triangle_points)
        M = cv2.getAffineTransform(landmark_triangle_points, mean_landmark_triangle_points)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle,
                                        mask=cropped_mean_landmark_triangle_mask)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle,
                                        mask=warped_image_mask[y:y+h,x:x+w])
        warped_image[y:y+h,x:x+w] += warped_triangle
        warped_image_mask[y:y+h,x:x+w][cropped_mean_landmark_triangle_mask==255] = 0

        if visualize:
            cv2.line(triangle_image, landmark_triangle_point_1.astype(int), landmark_triangle_point_2.astype(int), (0, 0, 255), 2)
            cv2.line(triangle_image, landmark_triangle_point_3.astype(int), landmark_triangle_point_2.astype(int), (0, 0, 255), 2)
            cv2.line(triangle_image, landmark_triangle_point_1.astype(int), landmark_triangle_point_3.astype(int), (0, 0, 255), 2)

    return warped_image, triangle_image
        