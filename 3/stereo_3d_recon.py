import numpy as np

from calibration import compute_mx_my, estimate_f_b
from extract_patches import extract_patches


def triangulate(u_left, u_right, v_left, calib_dict):
    """
    Triangulate (determine 3D world coordinates) a set of points given their projected coordinates in two images.
    These equations are according to the simple setup, where C' = (b, 0, 0)

    Args:
        u_left  (np.array of shape (num_points,))   ... Projected u-coordinates of the 3D-points in the left image
        u_right (np.array of shape (num_points,))   ... Projected u-coordinates of the 3D-points in the right image
        v       (np.array of shape (num_points,))   ... Projected v-coordinates of the 3D-points (same for both images)
        calib_dict (dict)                           ... Dict containing camera parameters
    
    Returns:
        points (np.array of shape (num_points, 3)   ... Triangulated 3D coordinates of the input - in units of [mm]
    """
    # Extract calibration parameters
    f = calib_dict['f']
    b = calib_dict['b']
    mx = calib_dict['mx']
    my = calib_dict['my']
    o_x = calib_dict['o_x']
    o_y = calib_dict['o_y']

    # Compute disparities
    disparities = u_left - u_right
    x, y, z = np.zeros_like(u_left), np.zeros_like(u_left), np.zeros_like(u_left)

    # Avoid division by zero
#    disparities[disparities == 0] = 1e6
    mask = (u_left != u_right)

    # Compute 3D coordinates
    z[mask] = f * b * mx / disparities[mask]
    x[mask] = (u_left[mask] - o_x) * z[mask] / (mx * f)
    y[mask] = (v_left[mask] - o_y) * z[mask] / (my * f)

    # Stack coordinates
    points = np.stack((x, y, z), axis=-1)

    return points


def compute_ncc(img_l, img_r, p):
    """
    Calculate normalized cross-correlation (NCC) between patches at the same row in two images.
    
    The regions near the boundary of the image, where the patches go out of image, are ignored.
    That is, for an input image, "p" number of rows and columns will be ignored on each side.

    For input images of size (H, W, C), the output will be an array of size (H - 2*p, W - 2*p, W - 2*p)

    Args:
        img_l (np.array of shape (H, W, C)) ... Left image
        img_r (np.array of shape (H, W, C)) ... Right image
        p (int):                            ... Defines square neighborhood. Patch-size is (2*p+1, 2*p+1).
                              
    Returns:
        corr    ... (np.array of shape (H - 2*p, W - 2*p, W - 2*p))
                    The value output[r, c_l, c_r] denotes the NCC between the patch centered at (r + p, c_l + p) 
                    in the left image and the patch centered at  (r + p, c_r + p) at the right image.
    """

    # Add dummy channel dimension
    if img_l.ndim == 2:
        img_l = img_l[:, :, None]
        img_r = img_r[:, :, None]
    
    assert img_l.ndim == 3, f"Expected 3 dimensional input. Got {img_l.shape}"
    assert img_l.shape == img_r.shape, "Shape mismatch."
    
    H, W, C = img_l.shape

    # Extract patches - patches_l/r are NumPy arrays of shape H, W, C * (2*p+1)**2
    patches_l = extract_patches(img_l, 2*p+1)
    patches_r = extract_patches(img_r, 2*p+1)
            
    # Reshape patches for matrix multiplication
    patches_l = patches_l.reshape(H, W, -1)
    patches_r = patches_r.reshape(H, W, -1)

    # Standardize each patch
    patches_l_mean = np.mean(patches_l, axis=-1, keepdims=True)
    patches_r_mean = np.mean(patches_r, axis=-1, keepdims=True)

    patches_l_std = np.std(patches_l, axis=-1, keepdims=True)
    patches_r_std = np.std(patches_r, axis=-1, keepdims=True)

    patches_l = (patches_l - patches_l_mean) / patches_l_std
    patches_r = (patches_r - patches_r_mean) / patches_r_std


    # Compute correlation using matrix multiplication

    corr = np.matmul(patches_l, patches_r.transpose(0, 2, 1))/((2*p+1)**2)

    # Ignore boundaries
    return corr[p:H-p, p:W-p, p:W-p]

class Stereo3dReconstructor:
    def __init__(self, p=5, w_mode='none'):
        """
        Feel free to add hyper parameters here, but be sure to set defaults
        
        Args:
            p       ... Patch size for NCC computation
            w_mode  ... Weighting mode. I.e. method to compute certainty scores
        """
        self.p = p
        self.w_mode = w_mode

    def fill_calib_dict(self, calib_dict, calib_points):
        """ Fill missing entries in calib dict - nothing to do here """
        calib_dict['mx'], calib_dict['my'] = compute_mx_my(calib_dict)
        calib_dict['f'], calib_dict['b'] = estimate_f_b(calib_dict, calib_points)
        
        return calib_dict

    def recon_scene_3d(self, img_l, img_r, calib_dict):
        """
        Compute point correspondences for two images and perform 3D reconstruction.

        Args:
            img_l (np.array of shape (H, W, C)) ... Left image
            img_r (np.array of shape (H, W, C)) ... Right image
            calib_dict (dict)                   ... Dict containing camera parameters
        
        Returns:
            points3d (np.array of shape (H, W, 4)
                Array containing the re-constructed 3D world coordinates for each pixel in the left image.
                Boundary points - which are not well defined for NCC might be padded with 0s.
                4th dimension holds the certainties.
        """

        # Add dummy channel dimension
        if img_l.ndim == 2:
            img_l = img_l[:, :, None]
            img_r = img_r[:, :, None]
        
        assert img_l.ndim == 3, f"Expected 3 dimensional input. Got {img_l.shape}"
        assert img_l.shape == img_r.shape, "Shape mismatch."
        
        H, W, C = img_l.shape
        
        # During NCC comutation we discard boundaries
        # This shifts the indices of the center pixels
        H_small, W_small = H - 2*self.p, W - 2*self.p
        
        calib_small = calib_dict.copy()
        calib_small['height'], calib_small['width'] = H_small, W_small
        calib_small['o_x'] = calib_dict['o_x'] - self.p
        calib_small['o_y'] = calib_dict['o_y'] - self.p
        
        # Create (u, v) pixel grid
        y, u_left = np.meshgrid(
            np.arange(H_small, dtype=float),
            np.arange(W_small, dtype=float),
            indexing='ij'
        )

        # Compute normalized cross correlation & find correspondece
        corr = compute_ncc(img_l, img_r, self.p)
        corr_corr = corr
        corr_corr[:, np.triu_indices(W_small, k=1)[0], np.triu_indices(W_small, k=1)[1]] = 0

        # Find best match correspondence
        u_right = np.argmax(corr_corr, axis=2)
                
        # Set certainty
        
        if self.w_mode == 'none':
            certainty_score = np.ones((H_small, W_small), dtype=float)
        else:
            # The certainty score is a measure of how confident we are in the correctness of the disparity estimation.
            # The score should be high where the NCC is high and low where the NCC is low.        
            certainty_score = np.max(corr_corr, axis=2)

            # Normalize the certainty score between zero and one for all pixels
            certainty_score = (certainty_score - np.min(certainty_score)) / (np.max(certainty_score) - np.min(certainty_score))
        
        # Triangulate the points to get 3D world coordinates
        points = triangulate(
            u_left.flatten(), u_right.flatten(), y.flatten(), calib_small
        )

        # Reshape to image & pad
        points = points.reshape(H_small, W_small, 3)
        points = np.pad(points, ((self.p, self.p), (self.p, self.p), (0, 0)))
        certainty_score = np.pad(certainty_score, self.p)[:, :, None]

        return np.concatenate([points, certainty_score], axis=2)