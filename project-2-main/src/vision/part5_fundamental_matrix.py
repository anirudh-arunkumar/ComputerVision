"""Fundamental matrix utilities."""

import numpy as np


def normalize_points(points: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Perform coordinate normalization through linear transformations.
    Args:
        points: A numpy array of shape (N, 2) representing the 2D points in
            the image

    Returns:
        points_normalized: A numpy array of shape (N, 2) representing the
            normalized 2D points in the image
        T: transformation matrix representing the product of the scale and
            offset matrices
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    # raise NotImplementedError(
    #     "`normalize_points` function in "
    #     + "`fundamental_matrix.py` needs to be implemented"
    # )
    central = np.mean(points, axis=0)
    shifted = points - central
    standard_dev = np.std(shifted, axis=0, ddof=0)
    
    x_scale = 1.0 / standard_dev[0]
    y_scale = 1.0 / standard_dev[1]
    T = np.array([[x_scale, 0, -1 * x_scale * central[0]], [0, y_scale, -1 * y_scale * central[1]], [0, 0, 1]])

    num_points = points.shape[0]
    points_homo = np.hstack([points, np.ones((num_points, 1))])
    normalized = (T @ points_homo.T).T
    points_normalized = normalized[:, :2] / normalized[:, 2][:, np.newaxis]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return points_normalized, T


def unnormalize_F(F_norm: np.ndarray, T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """
    Adjusts F to account for normalized coordinates by using the transformation
    matrices.

    Args:
        F_norm: A numpy array of shape (3, 3) representing the normalized
            fundamental matrix
        T_a: Transformation matrix for image A
        T_B: Transformation matrix for image B

    Returns:
        F_orig: A numpy array of shape (3, 3) representing the original
            fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    F_orig = T_b.T @ F_norm @ T_a

    # raise NotImplementedError(
    #     "`unnormalize_F` function in "
    #     + "`fundamental_matrix.py` needs to be implemented"
    # )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F_orig


def make_singular(F_norm: np.array) -> np.ndarray:
    """
    Force F to be singular by zeroing the smallest of its singular values.
    This is done because F is not supposed to be full rank, but an inaccurate
    solution may end up as rank 3.

    Args:
    - F_norm: A numpy array of shape (3,3) representing the normalized fundamental matrix.

    Returns:
    - F_norm_s: A numpy array of shape (3, 3) representing the normalized fundamental matrix
                with only rank 2.
    """
    U, D, Vt = np.linalg.svd(F_norm)
    D[-1] = 0
    F_norm_s = np.dot(np.dot(U, np.diag(D)), Vt)

    return F_norm_s


def estimate_fundamental_matrix(
    points_a: np.ndarray, points_b: np.ndarray
) -> np.ndarray:
    """
    Calculates the fundamental matrix. You may use the normalize_points() and
    unnormalize_F() functions here. Equation (9) in the documentation indicates
    one equation of a linear system in which you'll want to solve for f_{i, j}.

    Since the matrix is defined up to a scale, many solutions exist. To constrain
    your solution, use can either use SVD and use the last Vt vector as your
    solution, or you can fix f_{3, 3} to be 1 and solve with least squares.

    Be sure to reduce the rank of your estimate - it should be rank 2. The
    make_singular() function can do this for you.

    Args:
        points_a: A numpy array of shape (N, 2) representing the 2D points in
            image A
        points_b: A numpy array of shape (N, 2) representing the 2D points in
            image B

    Returns:
        F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    a_normal, T_a = normalize_points(points_a)
    b_normal, T_b = normalize_points(points_b)
    A = np.zeros((a_normal.shape[0], 9))

    for i in range(a_normal.shape[0]):
        u1, v1 = a_normal[i]
        u2, v2 = b_normal[i]

        A[i] = [u2 * u1, u2*v1, u2, v2 *u1, v2* v1, v2, u1, v1, 1]
    
    temp1, temp2, Vt = np.linalg.svd(A)
    f = Vt[-1]
    F_normal = f.reshape((3, 3))
    F_normal = make_singular(F_norm=F_normal)
    F = unnormalize_F(F_norm=F_normal, T_a=T_a, T_b=T_b)

    if not np.isclose(F[-1, -1], 0):
        F = F / F[-1, -1]

    # raise NotImplementedError(
    #     "`estimate_fundamental_matrix` function in "
    #     + "`fundamental_matrix.py` needs to be implemented"
    # )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F
