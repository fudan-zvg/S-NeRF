import cv2
from matplotlib.pyplot import sca
import numpy as np
from PIL import Image
from os.path import join
from math import sin, cos
from functools import reduce
import scipy.signal as signal

def conv2D(array,
           kernal,
           padding,
           dtype,
           smooth=False):
    '''
    Convolution.
    Args:
        array: Numpy array.
        kernal: Convolution kernal, a numpy array.
        padding: A two-dimensial tuple, the format is same as np.pad.
        dtype: The type of the resulting matrix.
        smooth: To modify the edge of the result (Only for smoothing).
    Returns:
        The result array.
    Note:
        Please input 2-dimensial array.
    '''
    
    assert len(array.shape) == 2 and len(kernal.shape) == 2
    
    array_new = np.pad(array, padding)
    h, w = array_new.shape
    s, _ = kernal.shape
    
    assert s % 2 == 1
    
    result_mat = np.zeros((h-s+1, w-s+1), dtype=dtype)
    for i in range((s-1)//2, h-(s-1)//2):
        for j in range((s-1)//2, w-(s-1)//2):
            result_mat[i-(s-1)//2, j-(s-1)//2] = np.dot(
                array_new[i-(s-1)//2 : i+(s-1)//2 + 1, j-(s-1)//2 : j+(s-1)//2 + 1].flatten(),
                kernal.flatten()[::-1]
                )
    
    if smooth:
        mask_mat = np.ones_like(array, dtype=np.uint8)
        mask_mat = np.pad(mask_mat, (padding))
        # Modify.
        for i in range((s-1)//2, h-(s-1)//2):
            for j in range((s-1)//2, w-(s-1)//2):   
                valid = np.sum(mask_mat[i-(s-1)//2 : i+(s-1)//2 + 1, j-(s-1)//2 : j+(s-1)//2 + 1])     
                result_mat[i-(s-1)//2, j-(s-1)//2] = result_mat[i-(s-1)//2, j-(s-1)//2] * (s**2) / valid
    
    return result_mat


def dilate(im_mat, r, dist_type):
    '''
    **Note**:
        Only support binary figures. 
    
    Dilation.
    Args:
        im_mat: An [h, w] array, representing an image.
        r: "Radius" of dilation. 
        dist_type: "Euclid", "D4" or "D8".
        pad_val: Padding value.
    Returns:
        An [h, w] array, representing the result binary image. (dtype=np.uint8)
    
    '''
    assert dist_type in ['Euclid', 'D4', 'D8']
    
    def distance(pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        if dist_type == 'Euclid':
            return np.sqrt((x1-x2) ** 2 + (y1-y2) ** 2)
        elif dist_type == 'D4':
            return np.abs(x1-x2) + np.abs(y1-y2)
        else:
            return np.max((np.abs(x1-x2), np.abs(y1-y2)))

    # Generate the kernal.
    kernal = np.zeros((2*r+1, 2*r+1), dtype=np.float32)
    center = (r, r)
    valid = [(x, y) for x in range(2*r+1) for y in range(2*r+1) if distance((x,y), center) <= r]
    for pos in valid:
        kernal[pos] = 1
    kernal = kernal / np.sum(kernal)    # Scale.
    # Conv.
    result_im = conv2D(array=im_mat,
                       kernal=kernal,
                       padding=r,
                       dtype=np.float32,
                       smooth=False)
    result_im = (result_im > 0)
    
    return result_im


def affine_trans_rev(im_mat, A):
    '''
    Affine transformation. (Backward approach)
    Args:
        im_mat: An array.
        A: The affine transformation matrix, with size of 3*3.
    '''
    # im_mat = np.array(im)
    im_mat = np.expand_dims(im_mat, 2) if len(im_mat.shape) == 2 else im_mat  # [h, w] -> [h, w, 1]
    h, w, channel = im_mat.shape
    
    trans_mat = A
    
    # Inverse.
    trans_mat_rev = np.linalg.inv(trans_mat)    
    
    target_im_mat = np.zeros_like(im_mat)
    
    # Get pixels' coordinates of target image. (Represent as a [3, N] matrix)
    target_im_mat = np.zeros_like(im_mat)
    target_coords = np.where(target_im_mat[..., 0] >= 0)   # Naive approach
    target_coords = np.vstack(target_coords)
    target_coords = np.vstack((target_coords, np.ones(h*w)))        
    pixels_total = target_coords.shape[1]  
    
    # Inverse transformation to get the source pixels.
    source_coords = np.matmul(trans_mat_rev, target_coords) 
    
    # Compute the "source" pixel value.
    for k in range(pixels_total):
        source_i, source_j, _ = source_coords[:, k]
        
        # Case 1: Interpolate.
        if source_i >= 0 and source_i < h-1 and source_j >= 0 and source_j < w-1:
            four_corners = (
                (int(source_i), int(source_j)),
                (int(source_i), int(source_j + 1)),
                (int(source_i + 1), int(source_j)),
                (int(source_i + 1), int(source_j + 1))
            )
            neighbour_pixels = [im_mat[idx] for idx in four_corners]
            # coefficients.
            coef = [abs((idx[0]-source_i) * (idx[1]-source_j)) for idx in four_corners]
            coef.reverse()
            
            pixel_val = []
            for c in range(channel):
                neighbour_pixels_val = [pixel[c] for pixel in neighbour_pixels]
                pixel_val.append(np.dot(np.array(neighbour_pixels_val), np.array(coef)))
            pixel_val = np.array(pixel_val)
            
        # Case 2: Extrapolate.
        else:
            if source_i < 0:
                if source_j < 0:
                    nearest_idx = (0, 0)
                elif source_j > w-1:
                    nearest_idx = (0, w-1)
                else:
                    nearest_idx = (0, round(source_j))
            elif source_i > h-1:
                if source_j < 0:
                    nearest_idx = (h-1, 0)
                elif source_j > w-1:
                    nearest_idx = (h-1, w-1)
                else:
                    nearest_idx = (h-1, round(source_j))
            else:
                if source_j < 0:
                    nearest_idx = (round(source_i), 0)
                else:
                    nearest_idx = (round(source_i), w-1)
                    
            pixel_val = im_mat[nearest_idx]
            
        # Get the target pixel value.
        target_i, target_j = round(target_coords[0, k]), round(target_coords[1, k])
        target_im_mat[target_i, target_j] = pixel_val
    
    if channel == 1:
        target_im_mat = target_im_mat[..., 0]
        
    return target_im_mat


def get_bound(mask_mat, r):
    # Bound.
    kernal = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])
    bound = signal.convolve2d(mask_mat, kernal, mode='same')
    bound = np.logical_and(bound < 9, bound > 0)    
    bound = dilate(bound, r, "Euclid")
    return bound


def transformation(im, 
                   mask_im,
                   coord_from,
                   coord_to,
                   depth_from,
                   depth_to):
    '''
    Translate a vehicle image.
    Args:
        im: An Image instance.
        mask_im: An Image instance, for mask.
        coord_from: (x_from, y_from)
        coord_to: (x_to, y_to)
    '''
    #### Affine transformation. ####
    im_mat = np.array(im)
    mask_im_mat = np.array(mask_im)
    mask_im_mat = np.expand_dims(mask_im_mat, axis=2).repeat(3, axis=2)   # [H, W] -> [H, W, 3]
    tx = coord_to[0] - coord_from[0]
    ty = coord_to[1] - coord_from[1]
    scale = depth_from / depth_to
    moving_matrix = np.float64([[scale,0,tx+coord_from[0]*(1-scale)], [0,scale,ty+coord_from[1]*(1-scale)]])
    rows, cols = im_mat.shape[:2]
    im_new_mat = cv2.warpAffine(im_mat, moving_matrix, (cols, rows))
    mask_im_mat = cv2.warpAffine(mask_im_mat, moving_matrix, (cols, rows))
    ################################
    
    #### Crop the vehicle's im. ####
    im_new_mat[np.where(mask_im_mat == 0)] = 0
    ################################
    
    return Image.fromarray(im_new_mat), Image.fromarray(mask_im_mat), scale


def scale(im,
          mask_im,
          depth_im,
          coord,
          scale):
    '''
    Scale to get the vehicle's image.
    Args:
        im: Image instance, representing the vehicle's raw image.
        mask_im: Image instance, representing the mask image.
        coord: The coord of scaling point.
        scale: ...
    '''
    
    #### Affine transformation. ####
    # First scale.
    im_mat = np.array(im)
    mask_im_mat = np.array(mask_im)
    depth_im_mat = np.array(depth_im)
    # mask_im_mat = np.expand_dims(mask_im_mat, axis=2).repeat(3, axis=2)   # [H, W] -> [H, W, 3]
    # scale_matrix = np.float64([[scale, 0, 0],
    #                            [0, scale, 0]])
    # rows, cols = im_mat.shape[:2]
    # im_scaled_mat = cv2.warpAffine(im_mat, scale_matrix, (cols, rows))
    # mask_im_scaled_mat = cv2.warpAffine(mask_im_mat, scale_matrix, (cols, rows))
    
    x, y = coord
    x_scaled, y_scaled = x*scale, y*scale
    dx = x - x_scaled
    dy = y - y_scaled
    trans_matrix = np.float64([[scale, 0, dx],
                              [0, scale, dy]])
    rows, cols = im_mat.shape[:2]
    im_final_mat = cv2.warpAffine(im_mat, trans_matrix, (cols, rows))
    mask_im_final_mat = cv2.warpAffine(mask_im_mat, trans_matrix, (cols, rows))
    depth_im_final_mat = cv2.warpAffine(depth_im_mat, trans_matrix, (cols, rows))
    ################################
    
    return Image.fromarray(im_final_mat), Image.fromarray(mask_im_final_mat), Image.fromarray(depth_im_final_mat)


def trans(vehicle_im,
          mask_im,
          depth_im,
          coords_from,
          coords_to):
    '''
    Translate the vehicle image (cropped).
    Args:
        vehicle_im: Image instance, representing the cropped vehicle's image.
        mask_im: Image instance representing the mask image.
        coords_from, coords_to.
    '''
    im_mat = np.array(vehicle_im)
    mask_im_mat = np.array(mask_im)
    depth_im_mat = np.array(depth_im)
    
    x, y = coords_from
    x_to, y_to = coords_to
    rows, cols = im_mat.shape[:2]
    
    dx = x_to - x
    dy = y_to - y
    
    trans_matrix = np.float64([[1, 0, dx],
                              [0, 1, dy]])
    im_final_mat = cv2.warpAffine(im_mat, trans_matrix, (cols, rows))
    mask_im_final_mat = cv2.warpAffine(mask_im_mat, trans_matrix, (cols, rows))
    depth_im_final_mat = cv2.warpAffine(depth_im_mat, trans_matrix, (cols, rows))
    
    return Image.fromarray(im_final_mat), Image.fromarray(mask_im_final_mat), Image.fromarray(depth_im_final_mat)



#### Refer to https://blog.csdn.net/qq_33854260/article/details/106297999 ####
def find_max_region(mask_sel):
    __, contours,hierarchy = cv2.findContours(mask_sel,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  
    area = []
 
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
 
    max_idx = np.argmax(area)
 
    max_area = cv2.contourArea(contours[max_idx])
 
    for k in range(len(contours)):
    
        if k != max_idx:
            cv2.fillPoly(mask_sel, [contours[k]], 0)
    return mask_sel