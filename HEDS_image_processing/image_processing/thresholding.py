import cv2
import numpy as np
from skimage.filters import threshold_otsu, threshold_multiotsu

def create_diff_image(img1, img2):
    """
    Takes two images and compares the differences between them, returning an image
    that represents the differences between the image, and another image which 
    shows the differences between them overlayed over the second image

    Parameters:
    - img1: grayscale image as a NumPy array with shape (256, 256, 1).
    - img2: grayscale image as a NumPy array with shape (256, 256, 1).
    
    Returns:
    - mask - binary image of differences between the two images
    - img2_with_diff - a modified version of img2 (shape: (256, 256, 1)) where the 
                        differences between it and img1 are shown in white.
    """
    # Find the pixel-wise differences between the two images
    diff = cv2.absdiff(img1, img2)

    # Create a mask for non-matching pixels (threshold for differences)
    threshold = 30  # You can adjust this threshold as needed
    mask = cv2.inRange(diff, threshold, 255)

    # Highlight the differences in white on the second image
    img2_with_diff = cv2.addWeighted(img2, 1, mask, 1, 0)
            
    return mask, img2_with_diff

def segment_two_phase(image):
    """
    Segments a grayscale image into two regions (black, gray,) based on Otsu thresholding

    Parameters:
    - image: grayscale image as a NumPy array with shape (256, 256, 1).
    
    Returns:
    - Segmented and colorized image as a NumPy array with shape (256, 256, 1).
    """
    # Calculate the Otsu threshold
    otsu_thresh = threshold_otsu(image)
    
    # Initialize the output image
    binary_image = np.zeros_like(image)
    
    # Apply the threshold - above threshold to gray, below threshold to black
    binary_image[image > otsu_thresh] = 128  # Set to gray
    # Pixels not meeting the above condition remain black by default
    
    return binary_image

def segment_three_phase(diff_image):
    """
    Segments a grayscale image into three regions (black, gray, white) based on Multi-Otsu thresholds 
    and assigns specific grayscale values to each region.
    
    Parameters:
    - image: diff'd grayscale image as a NumPy array with shape (256, 256, 1).
    
    Returns:
    - Segmented and colorized image as a NumPy array with shape (256, 256, 1).
    """
    # Calculate two thresholds using Multi-Otsu for three classes
    thresholds = threshold_multiotsu(diff_image, classes=3)
    
    # Initialize the output image with gray values first
    colorized_image = np.full(diff_image.shape, 128, dtype=np.uint8)  # Gray
    
    # Assign black to pixels <= the first threshold
    colorized_image[diff_image <= thresholds[0]] = 0  # Black
    
    # Assign white to pixels > the second threshold
    colorized_image[diff_image > thresholds[1]] = 255  # White
    
    return colorized_image

def compute_damage_fraction(thresh_img_three_phase):
    """
    Computes the damage (void + crack) percentage on a given image. 
    dmg_fraction = (dmg_pixels) / (dmg_pixels + crystal_pixels)
    
    Parameters:
    - thresh_img_three_phase: image that has beeen segmented into 3 phases 
       phases are: 0 - black/binder, 128 - gray/crystal, 255 - white/damage
    
    Returns:
    - float representing the damage fraction
    """

    dmg_count = np.sum(thresh_img_three_phase == 255) # count white damages
    crystal_count = np.sum(thresh_img_three_phase == 128) # count grey crystals
    fraction = dmg_count / (dmg_count + crystal_count)
    return fraction
