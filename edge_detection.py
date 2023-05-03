import cv2
import numpy as np
import scipy.io
import os
from matplotlib import pyplot as plt

test_path = r"BSDS500\data/images/test"
train_path = r"BSDS500\data\images\train"
validation_path = r"BSDS500\data\images\val"


def create_frames_from_video(video_Path, sobel_threshold_value = 100, ksize = 3):
    video = cv2.VideoCapture(video_Path)
    frames_list = []
    filtered_frames_list = []
    morph_frames_list = []

    while True:
        ret, frame = video.read()

        if ret:
            frame = cv2.resize(frame, (500, 500))
            frames_list.append(frame)
            binary_edge_map = perform_edge_detection(frame, ksize=ksize, threshold=sobel_threshold_value)
            filtered_frames_list.append(binary_edge_map)
        else:
            break
    video.release()
    return frames_list, filtered_frames_list#,morph_frames_list


def show_video_from_frames(frames_list, filtered_frames_list):
    for i in range(len(filtered_frames_list)):
        cv2.imshow("frame", frames_list[i])
        cv2.imshow("filtered_frames", filtered_frames_list[i])
        if(cv2.waitKey(5) == ord('q')):
            break
    cv2.destroyAllWindows()


def gray_scale(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def gaussian_blur(image):
    # Apply a Gaussian blur to the image
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    return blurred

def sobel_edge_detection(image, ksize=3, threshold=50):
    # Apply the Sobel edge detection algorithm
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = cv2.magnitude(sobelx, sobely)

    # Apply a threshold to the magnitude image
    thresholded = np.zeros_like(magnitude)
    thresholded[magnitude > threshold] = 255

    return thresholded


def morphological_operations(image):
    # Apply mathematical morphology operations
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(image, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    return erosion

def perform_edge_detection(image, ksize=3, threshold=50):
    # Convert the image to grayscale
    gray = gray_scale(image)

    # Apply a Gaussian blur to the image
    blurred = gaussian_blur(gray)

    # Apply the Sobel edge detection algorithm
    edges = sobel_edge_detection(blurred, ksize=ksize, threshold=threshold)

    # Apply mathematical morphology operations
    edges = morphological_operations(edges)
    return edges


def read_ground_truth_images(path, subscript=1):
    """
    Access the ground truth edge map for the first annotator
    gt_edge_map = gt_data['Boundaries'][0][0]

    Extract the numpy array from the list
    gt_edge_map = gt_edge_map.astype(float) / 255.0  # normalize to [0, 1]

    Do something with the ground truth edge map (e.g. evaluate an edge detector)
    """
    # Load the ground truth for a specific image
    gt_path = path
    gt_data = scipy.io.loadmat(gt_path)['groundTruth'][0]

    return gt_data[0][0][0][subscript]

def show_image(image, cmap = "gray", title = "Input Image"):
    plt.imshow(image, cmap=cmap)
    plt.title(title)#, plt.xticks([]), plt.yticks([])
    plt.show()

def read_images(path):
    return os.listdir(path)

def read_images_from_path(path):
    cv_images = []
    for image in read_images(path):
        image = cv2.imread(f"{test_path}/{image}")
        cv_images.append(image)
    return cv_images