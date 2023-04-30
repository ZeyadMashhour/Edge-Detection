import cv2
import numpy as np
import scipy.io
import os
from matplotlib import pyplot as plt

test_path = r"BSDS500\data/images/test"
train_path = r"BSDS500\data\images\train"
validation_path = r"BSDS500\data\images\val"


def create_frames_from_video(video_Path):
    video = cv2.VideoCapture(video_Path)
    frames_list = []
    filtered_frames_list = []
    morph_frames_list = []

    while True:
        ret, frame = video.read()

        if ret:
            # frame = cv2.resize(frame, (500, 500))
            frames_list.append(frame)
            binary_edge_map = create_sobel_filter_for_image(image=frame, x_kernel= 3, y_kernel=3, threshold_value=100)
            # laplacian = cv2.Laplacian(frame, cv2.CV_64F)
            # laplacian = np.uint8(laplacian)
            #filtered_frame = cv2.resize(binary_edge_map, (960, 540))

            morphImg = create_Mathematical_Morphology_for_image(frame)
            morphImg = create_sobel_filter_for_image(image=morphImg, x_kernel= 3, y_kernel=3, threshold_value=100)

            filtered_frames_list.append(binary_edge_map)
            morph_frames_list.append(morphImg)
        else:
            break
    video.release()
    return frames_list, filtered_frames_list,morph_frames_list


def show_video_from_frames(frames_list, filtered_frames_list,morph_frames_list):
    for i in range(len(filtered_frames_list)):
        cv2.imshow("frame", frames_list[i])
        cv2.imshow("filtered_frames", filtered_frames_list[i])
        cv2.imshow("morhped_frames", morph_frames_list[i])
        if(cv2.waitKey(5) == ord('q')):
            break
    cv2.destroyAllWindows()


def create_sobel_filter_for_image(image, x_kernel=3, y_kernel=3, threshold_value = 100):
    # convert image to gray scale 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = image
    # As the Kernel size increases, more pixels are now a part of the convolution process.
    # This signifies that the gradient map (edges) will tend to get blurry to a point the 
    # output looks likes a plastic cover has been wrapped around the edges.
    
    # define kernel for filter
    # Calculate gradient using Sobel kernels
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=x_kernel)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=y_kernel)

    # Calculate gradient magnitude
    grad_mag = cv2.magnitude(grad_x, grad_y)
     
    # reducing the threshold increases the edges
    # Apply thresholding to the gradient magnitude image
    _, binary_edge_map = cv2.threshold(grad_mag, threshold_value, 255, cv2.THRESH_BINARY)
    return np.uint8(binary_edge_map)

def create_Mathematical_Morphology_for_image(img):
    # Define a kernel for morphological operations
    kernel = np.ones((5,5), np.uint8)
    # Apply morphological closing to fill in small gaps in the edges
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # Apply morphological opening to remove small objects and noise from the edges
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    return opening

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