"""
metrics.py

Contains similarity and quality evaluation metrics used for
assessing mosaic accuracy. Includes MSE and SSIM.
"""

import numpy as np
from .image_processor import preprocess_image


# ---------------------------------------------------------
# Mean Squared Error (MSE)
# ---------------------------------------------------------

def compute_mse(original, mosaic) -> float:
    """
    Compute Mean Squared Error between the original image
    and the reconstructed mosaic.

    Parameters
    ----------
    original : np.ndarray or PIL.Image
        Original input image.

    mosaic : np.ndarray
        Output mosaic (H,W,3)

    Returns
    -------
    float
        MSE value.
    """
    original_proc = preprocess_image(original)
    mosaic = mosaic.astype(float)
    orig = original_proc.astype(float)

    return float(np.mean((orig - mosaic) ** 2))


# ---------------------------------------------------------
# Structural Similarity Index (SSIM)
# ---------------------------------------------------------

def _compute_ssim_channel(x, y):
    """
    Compute SSIM for one color channel.

    Uses the classic SSIM formula:
    SSIM = ((2μxμy + C1)(2σxy + C2)) / ((μx² + μy² + C1)(σx² + σy² + C2))

    This implementation is window-free and global,
    which is acceptable for this assignment.
    """

    # Means
    mu_x = x.mean()
    mu_y = y.mean()

    # Variances
    sigma_x = x.var()
    sigma_y = y.var()

    # Covariance
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    # Stability constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    return numerator / denominator


def compute_ssim(original, mosaic) -> float:
    """
    Compute SSIM (Structural Similarity Index) across RGB channels.

    Parameters
    ----------
    original : np.ndarray or PIL.Image
        Original input image.

    mosaic : np.ndarray
        Mosaic image.

    Returns
    -------
    float
        SSIM value in range [-1,1], where 1 = perfect match.
    """

    original_proc = preprocess_image(original).astype(float)
    mosaic = mosaic.astype(float)

    ssim_vals = []
    for ch in range(3):
        ssim_vals.append(
            _compute_ssim_channel(original_proc[:, :, ch], mosaic[:, :, ch])
        )

    # Average over RGB channels
    return float(np.mean(ssim_vals))


# ---------------------------------------------------------
# Convenience Wrapper
# ---------------------------------------------------------

def compute_metrics(original, mosaic):
    """
    Compute both MSE and SSIM.

    Returns
    -------
    dict
        {
            "mse": float,
            "ssim": float
        }
    """
    return {
        "mse": compute_mse(original, mosaic),
        "ssim": compute_ssim(original, mosaic),
    }