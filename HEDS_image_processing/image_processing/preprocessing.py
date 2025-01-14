import cv2
import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageOps, ImageEnhance


def remove_shadows(img): 
  rgb_planes = cv2.split(img)
  result_planes = []
  result_norm_planes = []
  for plane in rgb_planes:
      dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
      pil_img = Image.fromarray(dilated_img)
      bg_img = cv2.medianBlur(dilated_img, 21)
      pil_img = Image.fromarray(bg_img)

      diff_img = 255 - cv2.absdiff(plane, bg_img)
      pil_img = Image.fromarray(diff_img)
      norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
      result_planes.append(diff_img)
      result_norm_planes.append(norm_img)

  result = cv2.merge(result_planes)
  result_norm = cv2.merge(result_norm_planes)
  result = Image.fromarray(result)
  return result


def enhance_image(image, brightness_factor=0.7, contrast_factor=4, sharpness_factor=2):
  
  contraster = ImageEnhance.Contrast(image)
  result = contraster.enhance(contrast_factor)
  brightener = ImageEnhance.Brightness(result)
  result = brightener.enhance(brightness_factor)
  sharpener = ImageEnhance.Sharpness(result)
  result = sharpener.enhance(sharpness_factor)

  return result

def remove_islands(im, minsize = 30):
     thresh = sitk.BinaryThreshold(im,0,128,0,255)
     connect = sitk.ConnectedComponent(thresh, True)
     relabel = sitk.RelabelComponent(connect, minsize)
     largest = sitk.BinaryThreshold(relabel,1,255)

     return largest

def overlay(img, mask):
    # Create a copy of the img to avoid modifying the original image
    data = img.copy()
    # Apply mask: Set pixels in data to 0 where mask is 0
    data[mask == 0] = 0
    return data

def preprocess_image(img_path):
    """
    Takes an image path and preprocesses it for HEDS. 
    
    Parameters:
    - img_path: str corresponding an image path
    
    Returns:
    - overlayed - (w, h) np.ndarray image 
    """

    img_pil = Image.open(img_path)
    img_np = np.array(img_pil)
    shadowless_img = remove_shadows(img_np)
    enhanced_img_pil = enhance_image(shadowless_img)
    enhanced_grey_pil = enhanced_img_pil.convert('L')
    enhanced_grey_pil = enhanced_grey_pil.resize((256,256))
    enhanced_grey_np = np.array(enhanced_grey_pil)
    sitk_im = sitk.GetImageFromArray(enhanced_grey_np)
    removed_islands = remove_islands(sitk_im)
    np_im = sitk.GetArrayFromImage(removed_islands)
    overlayed = overlay(enhanced_grey_np, np_im)

    return overlayed






