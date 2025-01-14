import os
import numpy as np
from time import time
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from PIL import Image as im
from skimage.filters import threshold_otsu, threshold_multiotsu
from . import thresholding


def decorator_timer(func):
    # This function shows the execution time of the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

def normalize_0_255_to_0_1(arr):
    normalized_arr = arr / 255.0
    return normalized_arr

def denormalize_0_1_to_0_255_uint8(arr):
    img_arr = (arr * 255.0).astype('uint8')
    return img_arr

def normalize_0_255_to_neg1_1(arr):
    # Normalize the array to range [0, 1]
    normalized_arr = arr / 255.0
    # Scale and shift the normalized array to range [1, -1]
    rescaled_arr = 2 * normalized_arr - 1
    return rescaled_arr

def denormalize_neg1_1_to_0_255_uint8(arr):
    # Shift from [-1, 1] to [0, 2] and scale to [0, 1]
    shifted_scaled_arr = ((arr + 1) / 2)
    # Convert to [0, 255]
    denormalized_arr = shifted_scaled_arr * 255.0
    # round to the nearest integer and ensure the type is uint8 for image data
    denormalized_arr = np.round(denormalized_arr).astype(np.uint8)
    return denormalized_arr

@decorator_timer
def read_paths_and_threshold(img_paths, mask_paths, imsize=256, img_seg_type=0, mask_seg_type=0): 
    """
    Takes two lists/arrays (img_paths, mask_paths), reads them into images, 
    and then converts them into grayscale (256,256,1) images and then segments them 
    according to the segmentation type

    Parameters:
    - img_paths: list of strings for images to be read
    - mask_paths: list of strings for masks to be read
    - img_seg_type: type of segmentation to do on read images (0: none, 2: 2-phase)
    - mask_seg_type: type of segmentation to do on read masks (0: none, 2: 2-phase, 3: 3-phase)

    
    Returns:
    - images - np.ndarray of (n, 256, 256, 1) images where n is the amount of strings in the input list
    - masks - np.ndarray of (n, 256, 256, 1) images where n is the amount of strings in the input list
    """
    images = []
    masks = []
    for i, img in enumerate(img_paths): 
        img = im.open(img)
        grey_img = img.convert('L')
        grey_img = grey_img.resize((imsize,imsize))
        np_img = np.expand_dims(grey_img, axis=-1)
        if img_seg_type == 2:
            thresh_img = thresholding.segment_two_phase(np_img)
            images.append(thresh_img)
        else:
            images.append(np_img)
    
    for i, mask in enumerate(mask_paths): 
        mask = im.open(mask)
        grey_mask = mask.convert('L')
        grey_mask = grey_mask.resize((imsize,imsize))
        np_mask = np.expand_dims(grey_mask, axis=-1)
        if mask_seg_type == 2:
            thresh_mask = thresholding.segment_two_phase(np_mask)
            masks.append(thresh_mask)
        elif mask_seg_type == 3:
            diff = thresholding.create_diff_image(images[i], np_mask)
            thresh_mask = thresholding.segment_three_phase(diff)
            masks.append(thresh_mask)
        else:
            masks.append(np_mask)

    return np.array(images), np.array(masks) 


def load_img_paths(source_dir):
    img_paths = sorted( 
        [
            os.path.join(source_dir, fname)
            for fname in os.listdir(source_dir)
            if "groundtruth_" not in fname
        ]
    )
    
    mask_paths = sorted( 
        [
            os.path.join(source_dir, fname)
            for fname in os.listdir(source_dir)
            if "groundtruth_" in fname
        ]
    ) 
    return img_paths, mask_paths


@decorator_timer
def setup_datasets(A, B, norm_type='00', split_fraction=0.2):
    A_set = A 
    B_set = B 
    if norm_type == '01':
        A_set = normalize_0_255_to_0_1(A)
        B_set = normalize_0_255_to_0_1(B)
    if norm_type == '-11':
        A_set = normalize_0_255_to_neg1_1(A)
        B_set = normalize_0_255_to_neg1_1(B)

    train_A, val_A, train_B, val_B = train_test_split(A_set, B_set,
                                                test_size=split_fraction,
                                                random_state=0
                                                )
    
    return train_A, val_A, train_B, val_B