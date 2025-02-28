import math

import numpy as np
from vision.part5_fundamental_matrix import estimate_fundamental_matrix


def calculate_num_ransac_iterations(
    prob_success: float, sample_size: int, ind_prob_correct: float
) -> int:
    """
    Calculates the number of RANSAC iterations needed for a given guarantee of
    success.

    Args:
        prob_success: float representing the desired guarantee of success
        sample_size: int the number of samples included in each RANSAC
            iteration
        ind_prob_success: float representing the probability that each element
            in a sample is correct

    Returns:
        num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    # raise NotImplementedError(
    #     "`calculate_num_ransac_iterations` function "
    #     + "in `ransac.py` needs to be implemented"
    # )

    correct = ind_prob_correct ** sample_size

    if correct == 0:
        return float('inf')
    
    num_samples = math.log(1 - prob_success) / math.log(1 - correct)
    num_samples = int(math.ceil(num_samples))

    print(f"Sample Size: {sample_size}, Required RANSAC Iterations: {num_samples}")
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return int(num_samples)


def ransac_fundamental_matrix(
    matches_a: np.ndarray, matches_b: np.ndarray
) -> np.ndarray:
    """
    For this section, use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You would reuse
    estimate_fundamental_matrix() from part 2 of this assignment and
    calculate_num_ransac_iterations().

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 30 points for either left or
    right images.

    Tips:
        0. You will need to determine your prob_success, sample_size, and
            ind_prob_success values. What is an acceptable rate of success? How
            many points do you want to sample? What is your estimate of the
            correspondence accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for creating
            your random samples.
        2. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 0.1.

    Args:
        matches_a: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image A
        matches_b: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
        best_F: A numpy array of shape (3, 3) representing the best fundamental
            matrix estimation
        inliers_a: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image A that are inliers with respect to
            best_F
        inliers_b: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image B that are inliers with respect to
            best_F
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################



    # raise NotImplementedError(
    #     "`ransac_fundamental_matrix` function in "
    #     + "`ransac.py` needs to be implemented"
    # )

    thres = 0.1
    sample_size = 9
    succes = 0.99
    correct = 0.5
    total_matches = matches_a.shape[0]
    num_samples = calculate_num_ransac_iterations(prob_success=succes, sample_size=sample_size, ind_prob_correct=correct)
    homo_a = np.hstack([matches_a, np.ones((total_matches, 1))])
    homo_b = np.hstack([matches_b, np.ones((total_matches, 1))])
    inlier_c =0
    inlier_idx = None
    F_best = None

    for i in range(num_samples):
        idx = np.random.choice(total_matches, sample_size, replace=False)
        a_sample = matches_a[idx]
        b_sample = matches_b[idx]

        try:
            candidate = estimate_fundamental_matrix(a_sample, b_sample)
        except Exception:
            continue

        epi_lines = (candidate @ homo_a.T).T
        err = np.abs(np.sum(homo_b * epi_lines, axis=1)) / np.sqrt(epi_lines[:, 0]**2 + epi_lines[:, 1]**2 + 1e-8)
        indexes = np.where(err < thres)[0]
        count = len(indexes)

        if count > inlier_c:
            inlier_c = count
            F_best = candidate
            inlier_idx = indexes


    if F_best is None or inlier_idx is None:
        F_best = np.eye(3)
        a_inliers = np.array([])
        b_inliers = np.array([])
    else:
        a_inliers = matches_a[inlier_idx]
        b_inliers = matches_b[inlier_idx] 

    best_F = F_best
    inliers_a = a_inliers
    inliers_b = b_inliers

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return best_F, inliers_a, inliers_b
