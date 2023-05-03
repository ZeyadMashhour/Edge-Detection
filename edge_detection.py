import cv2  # Import the OpenCV library for computer vision and image processing
import numpy as np  # Import the NumPy library for numerical computations on arrays
import scipy.io  # Import the SciPy library for scientific computing and file input/output
import os  # Import the OS module for interacting with the operating system
from matplotlib import pyplot as plt  # Import the Pyplot module from the Matplotlib library for data visualization


test_path = r"BSDS500\data/images/test"  # Define the path to the test images directory
train_path = r"BSDS500\data\images\train"  # Define the path to the training images directory
validation_path = r"BSDS500\data\images\val"  # Define the path to the validation images directory



def create_frames_from_video(video_Path, sobel_threshold_value = 150, ksize = 3):
    # Open the video file
    video = cv2.VideoCapture(video_Path)

    # Initialize empty lists to store the frames and their filtered versions
    frames_list = []
    filtered_frames_list = []
    morph_frames_list = []

    # Read frames from the video until there are none left
    while True:
        ret, frame = video.read()

        # If there are still frames left, resize the frame and add it to the list
        if ret:
            frame = cv2.resize(frame, (500, 500))
            frames_list.append(frame)

            # Apply edge detection to the frame and add the resulting binary edge map to the list
            binary_edge_map = perform_edge_detection(frame, ksize=ksize, threshold=sobel_threshold_value)
            filtered_frames_list.append(binary_edge_map)

        # If there are no more frames, release the video object and return the lists
        else:
            video.release()
            return frames_list, filtered_frames_list#,morph_frames_list



def show_video_from_frames(frames_list, filtered_frames_list):
    # Loop through each frame and its filtered version
    for i in range(len(filtered_frames_list)):

        # Show the original frame and the filtered frame side by side
        cv2.imshow("frame", frames_list[i])
        cv2.imshow("filtered_frames", filtered_frames_list[i])

        # Wait for a key press and exit if the 'q' key is pressed
        if(cv2.waitKey(5) == ord('q')):
            break

    # Destroy all open windows
    cv2.destroyAllWindows()



def gray_scale(image):
    # Convert the image to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
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
    kernel = np.ones((3, 3), np.uint8)  # Create a 3x3 matrix of ones as a kernel
    dilation = cv2.dilate(image, kernel, iterations=1)  # Dilate the image using the kernel
    erosion = cv2.erode(dilation, kernel, iterations=1)  # Erode the dilated image using the kernel
    return erosion  # Return the eroded image


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

    return gt_data[0][0][0][subscript]  # Return the ground truth edge map as a numpy array


def show_image(image, cmap="gray", title="Input Image"):
    plt.imshow(image, cmap=cmap)  # Display the input image using matplotlib
    plt.title(title)  # Set the title of the displayed image
    plt.show()  # Display the image


def read_images(path):
    return os.listdir(path)  # Return a list of file names in the specified directory path


def read_images_from_path(path):
    cv_images = []  # Initialize an empty list to store the images
    for image in read_images(path):  # Loop through each image file in the directory
        image = cv2.imread(f"{test_path}/{image}")  # Read the image file using OpenCV
        cv_images.append(image)  # Append the image to the list of images
    return cv_images  # Return the list of OpenCV images
