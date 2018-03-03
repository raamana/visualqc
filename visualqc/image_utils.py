"""

Image processing utilities

"""

import numpy as np
from scipy import ndimage

def background_mask(mri, thresh_perc=1):
    """Creates the background mask from an MRI"""

    grad_magnitude = gradient_magnitude(mri)
    nonzero_grad_mag = grad_magnitude[grad_magnitude>0]

    thresh_val = np.percentile(nonzero_grad_mag.flatten(), thresh_perc)
    background_mask = grad_magnitude < thresh_val

    se36 = ndimage.generate_binary_structure(3, 6)
    closed = ndimage.binary_closing(background_mask, se36, iterations=6)
    final_mask = ndimage.binary_erosion(closed, se36, iterations=5)

    return final_mask


def gradient_magnitude(mri):
    """Computes the gradient magnitude"""

    grad = np.asarray(np.gradient(mri))
    grad_magnitude = np.sqrt(np.sum(np.power(grad, 2.), axis=0))

    return grad_magnitude
