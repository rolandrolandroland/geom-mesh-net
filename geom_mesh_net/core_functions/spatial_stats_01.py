from scipy.spatial.distance import pdist
import numpy as np
import math
rng = np.random.default_rng(42)


def calculate_spatial_barcode(coords_dict, bins=5, r_max=15.0, sample_size=500):
    """
    Generates a fast macroscopic 'barcode' of the point cloud geometry.
    Approximates Ripley's K-function by binning pairwise distances.

    Args: coords_dict (dict): A dictionary containing the point cloud coordinates.
                             Must contain the keys "x", "y", "z", which map to
                             arrays or lists of point locations
          bins (int, optional): The number of bins to use in the histogram
          r_max (float, optional): The maximum distance threshold to consider
                                   Any pair further apart than this won't be considered
          sample_size (int, optional): The number of points to sample before running the pairwise math
                                        because pdist calculates distances between EVERY PAIR (an O(N^2) operation),
                                        dropping this greatly reduces runtime

    Returns:
        np.ndarray: a 1D array of length 'bins' representing the normalized spatial signature,
        cast to float32 so it can be ingested directly by pytorch

    """
    # 1. Stack coordinates
    pts = np.vstack([coords_dict['x'], coords_dict['y'], coords_dict['z']]).T

    # 2. Subsample for speed (DataLoader must be fast)
    if len(pts) > sample_size:
        indices = rng.choice(len(pts), size=sample_size, replace=False)
        pts = pts[indices]

    # 3. Calculate all pairwise distances in the sample
    distances = pdist(pts)

    # 4. Bin the distances to create a histogram signature
    hist, _ = np.histogram(distances, bins=bins, range=(0, r_max))

    # 5. Normalize so the network can digest it cleanly (values between 0 and 1)
    if hist.max() > 0:
        hist = hist / hist.max()

    return hist.astype(np.float32)


def calculate_iou(preds, targets, threshold=0.5):
    """

    :param preds: predictions
    :param targets: targets (ground truth)
    :param threshold: threshold to determine intersection over union (cluster or not cluster)
    :return: intersection over union value for benchmarking
    """
    # each voxel of predictions and targets is assigned to be clustered or not
    # depending on if greater than threshold
    pred_mask = (preds > threshold).float()
    target_mask = (targets > threshold).float()

    intersection = (pred_mask * target_mask).sum()
    union = pred_mask.sum() + target_mask.sum() - intersection

    return (intersection / (union + 1e-8)).item()